import os

from src.losses.adversarial import NonSaturatingWithR1
from src.losses.feature_matching import masked_l1_loss, feature_matching_loss
from src.losses.perceptual import ResNetPL
from src.models.layers import GateConv
from src.models.LaMa import *
from src.models.TSR_model import *
from src.models.upsample import StructureUpsampling
from src.utils import get_lr_schedule_with_warmup, torch_init_model


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


class LaMaBaseInpaintingTrainingModule(nn.Module):
    def __init__(self, config, gpu, name, rank, *args, test=False, **kwargs):
        super().__init__(*args, **kwargs)
        print('BaseInpaintingTrainingModule init called')
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        self.generator = LaMa_model().cuda(gpu)
        self.best = None

        if not test:
            self.discriminator = NLayerDiscriminator(**self.config.discriminator).cuda(gpu)
            self.adversarial_loss = NonSaturatingWithR1(**self.config.losses['adversarial'])
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

        self.load()
        if self.config.DDP:
            import apex
            self.generator = apex.parallel.convert_syncbn_model(self.generator)
            self.discriminator = apex.parallel.convert_syncbn_model(self.discriminator)
            self.generator = apex.parallel.DistributedDataParallel(self.generator)
            self.discriminator = apex.parallel.DistributedDataParallel(self.discriminator)

    def load(self):
        if self.test:
            self.gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])

        if not self.test and os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.iteration = data['iteration']
            if self.iteration > 0:
                gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
                if torch.cuda.is_available():
                    data = torch.load(gen_weights_path)
                else:
                    data = torch.load(gen_weights_path, map_location=lambda storage, loc: storage)
                self.best = data['best_fid']
                print('Loading best psnr...')

        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if not self.test and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)
            self.dis_optimizer.load_state_dict(data['optimizer'])
            self.discriminator.load_state_dict(data['discriminator'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def save(self):
        print('\nsaving %s...\n' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]


class BaseInpaintingTrainingModule(nn.Module):
    def __init__(self, config, gpu, name, rank, *args, test=False, **kwargs):
        super().__init__(*args, **kwargs)
        print('BaseInpaintingTrainingModule init called')
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        self.generator = ReZeroFFC(config).cuda(gpu)
        self.best = None

        print('Loading %s StructureUpsampling...' % self.name)
        
        print("Loading trained transformer...")
     
        model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=256, block_size=32,
                                         attn_pdrop=0.0, n_layer=16, n_head=8)
        self.transformer = EdgeLineGPT256RelBCE(model_config)
        checkpoint = torch.load(config.transformer_ckpt_path, map_location='cpu')
        if config.transformer_ckpt_path.endswith('.pt'):
            self.transformer.load_state_dict(checkpoint)
        else:
            self.transformer.load_state_dict(checkpoint['model'])
        self.transformer.cuda(gpu).eval()
        self.transformer.half()

        if not test:
            self.discriminator = NLayerDiscriminator(**self.config.discriminator).cuda(gpu)
            self.adversarial_loss = NonSaturatingWithR1(**self.config.losses['adversarial'])
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
            
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()
        if not test:
            self.load_rezero()  # load pretrain model
        self.load()  # reload for restore

        # reset lr
        if not test:
            for group in self.gen_optimizer.param_groups:
                group['lr'] = config.optimizers['generator']['lr']
                group['initial_lr'] = config.optimizers['generator']['lr']
            for group in self.dis_optimizer.param_groups:
                group['lr'] = config.optimizers['discriminator']['lr']
                group['initial_lr'] = config.optimizers['discriminator']['lr']

        if self.config.DDP and not test:
            import apex
            self.generator = apex.parallel.convert_syncbn_model(self.generator)
            self.discriminator = apex.parallel.convert_syncbn_model(self.discriminator)
            self.generator = apex.parallel.DistributedDataParallel(self.generator)
            self.discriminator = apex.parallel.DistributedDataParallel(self.discriminator)

        if self.config.optimizers['decay_steps'] is not None and self.config.optimizers['decay_steps'] > 0 and not test:
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.dis_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
                                                   
            if self.iteration - self.config.START_ITERS > 1:
                for _ in range(self.iteration - self.config.START_ITERS):
                    self.g_scheduler.step()
                    self.d_scheduler.step()
                   
        else:
            self.g_scheduler = None
            self.d_scheduler = None
            

    def load_rezero(self):
        if os.path.exists(self.config.gen_weights_path0):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.config.gen_weights_path0, map_location='cpu')
            torch_init_model(self.generator, data, 'generator', rank=self.global_rank)
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.iteration = data['iteration']
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if (self.config.MODE == 1 or self.config.score) and os.path.exists(self.config.dis_weights_path0):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.config.dis_weights_path0, map_location='cpu')
            torch_init_model(self.discriminator, data, 'discriminator', rank=self.global_rank)
            self.dis_optimizer.load_state_dict(data['optimizer'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def load(self):
        if self.test:
            self.gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
          
        if not self.test and os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
            
            self.gen_optimizer.load_state_dict(data['optimizer'])
           
            self.iteration = data['iteration']
            if self.iteration > 0:
                gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen_HR.pth')
                data = torch.load(gen_weights_path, map_location='cpu')
                self.best = data['best_fid']
                print('Loading best fid...')

        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if not self.test and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.dis_weights_path, map_location='cpu')
            self.dis_optimizer.load_state_dict(data['optimizer'])
            self.discriminator.load_state_dict(data['discriminator'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def save(self):
        print('\nsaving %s...\n' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
       
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]


class LaMaInpaintingTrainingModule(LaMaBaseInpaintingTrainingModule):
    def __init__(self, *args, gpu, rank, image_to_discriminator='predicted_image', test=False, **kwargs):
        super().__init__(*args, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.image_to_discriminator = image_to_discriminator
        self.refine_mask_for_losses = None

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        batch['predicted_image'] = self.generator(masked_img.to(torch.float32),mask)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        self.iteration += 1

        self.discriminator.zero_grad()
        # discriminator loss
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])

        real_loss, _, _ = self.adversarial_loss.discriminator_real_loss(real_batch=batch['image'],
                                                                  discr_real_pred=discr_real_pred)
        batch = self.forward(batch)
        predicted_img = batch[self.image_to_discriminator].detach()

        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))

        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['mask'])

        dis_loss = fake_loss + real_loss

        dis_metric = {}
        dis_metric['discr_adv'] = dis_loss.item()
        dis_metric.update(add_prefix_to_keys(dis_metric, 'adv_'))

        dis_loss.backward()
        self.dis_optimizer.step()

        # generator loss
        self.generator.zero_grad()
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses['l1']['weight_known'],
                                  self.config.losses['l1']['weight_missing'])

        gen_loss = l1_value
        gen_metric = dict(gen_l1=l1_value.item())

        # vgg-based perceptual loss
        if self.config.losses['perceptual']['weight'] > 0:
            pl_value = self.loss_pl(predicted_img, img,
                                    mask=supervised_mask).sum() * self.config.losses['perceptual']['weight']
            gen_loss = gen_loss + pl_value
            gen_metric['gen_pl'] = pl_value.item()

        
        mask_for_discr = original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        gen_loss = gen_loss + adv_gen_loss
        gen_metric['gen_adv'] = adv_gen_loss.item()
        gen_metric.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            discr_real_pred, discr_real_features = self.discriminator(img)
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            gen_loss = gen_loss + fm_value
            gen_metric['gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            with torch.cuda.amp.autocast():
                resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            gen_loss = gen_loss + resnet_pl_value
            gen_metric['gen_resnet_pl'] = resnet_pl_value.item()

        if self.config.AMP:
            self.scaler.scale(gen_loss).backward()
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
            gen_metric['loss_scale'] = self.scaler.get_scale()
        else:
            gen_loss.backward()
            self.gen_optimizer.step()
        # create logs
        logs = [dis_metric, gen_metric]

        return batch['predicted_image'], gen_loss, dis_loss, logs, batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, gpu, rank, image_to_discriminator='predicted_image', test=False, **kwargs):
        super().__init__(*args, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.image_to_discriminator = image_to_discriminator
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = GateConv(in_channels=2, out_channels=64, kernel_size=7, stride=1,padding=0)

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)
       
        masked_img = torch.cat([masked_img, mask], dim=1)
        masked_str = torch.cat([batch['edge'], batch['line']], dim=1)
        masked_str = self.conv1(self.pad1(masked_str))
      
        batch['predicted_image'] = self.generator(masked_img.to(torch.float32),mask.to(torch.float32), batch['rel_pos'],
                                                      batch['direct'],masked_str.to(torch.float32))
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        self.iteration += 1
        self.discriminator.zero_grad()
        # discriminator loss
        dis_loss, batch, dis_metric = self.discriminator_loss(batch)
        self.dis_optimizer.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()

        # generator loss
        self.generator.zero_grad()
        
        gen_loss, gen_metric = self.generator_loss(batch)

        if self.config.AMP:
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
          
        else:
            self.gen_optimizer.step()
           
        if self.g_scheduler is not None:
            self.g_scheduler.step()

        # create logs
        if self.config.AMP:
            gen_metric['loss_scale'] = self.scaler.get_scale()
        logs = [dis_metric, gen_metric]

        return batch['predicted_image'], gen_loss, dis_loss, logs, batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses['l1']['weight_known'],
                                  self.config.losses['l1']['weight_missing'])

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value.item())

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss.item()
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))
        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            discr_real_pred, discr_real_features = self.discriminator(img)
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value.item()

        if self.config.AMP:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        return total_loss.item(), metrics

    def discriminator_loss(self, batch):
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        real_loss, dis_real_loss, grad_penalty = self.adversarial_loss.discriminator_real_loss(
            real_batch=batch['image'],
            discr_real_pred=discr_real_pred)
        real_loss.backward()
        if self.config.AMP:
            with torch.cuda.amp.autocast():
                batch = self.forward(batch)
        else:
            batch = self(batch)
        batch[self.image_to_discriminator] = batch[self.image_to_discriminator].to(torch.float32)
        predicted_img = batch[self.image_to_discriminator].detach()
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['mask'])
        fake_loss.backward()
        total_loss = fake_loss + real_loss
        metrics = {}
        metrics['dis_real_loss'] = dis_real_loss.mean().item()
        metrics['dis_fake_loss'] = fake_loss.item()
        metrics['grad_penalty'] = grad_penalty.mean().item()

        return total_loss.item(), batch, metrics
    
    def SampleEdgeLineLogits(img_idx, edge_idx, line_idx,mask=None, iterations=1, device='cuda', add_v=0.05, mul_v=4):
        # [img, edge, line] = context
        img = img_idx.to(device)
        edge = edge_idx.to(device)
        line = line_idx.to(device)
        mask = mask.to(device)
        img = img * (1 - mask)
        edge = edge * (1 - mask)
        line = line * (1 - mask)
        model.eval()
        with torch.no_grad():
            for i in range(iterations):
                edge_logits, line_logits = self.transformer.forward_with_logits(img.to(torch.float16),
                                                                     edge.to(torch.float16),
                                                                     line.to(torch.float16),
                                                                     masks=mask.to(torch.float16))
                edge_pred = torch.sigmoid(edge_logits)
                line_pred = torch.sigmoid((line_logits + add_v) * mul_v)
                edge = edge + edge_pred * mask
                edge[edge >= 0.25] = 1
                edge[edge < 0.25] = 0
                line = line + line_pred * mask

                b, _, h, w = edge_pred.shape
                edge_pred = edge_pred.reshape(b, -1, 1)
                line_pred = line_pred.reshape(b, -1, 1)
                mask = mask.reshape(b, -1)

                edge_probs = torch.cat([1 - edge_pred, edge_pred], dim=-1)
                line_probs = torch.cat([1 - line_pred, line_pred], dim=-1)
                edge_probs[:, :, 1] += 0.5
                line_probs[:, :, 1] += 0.5
                edge_max_probs = edge_probs.max(dim=-1)[0] + (1 - mask) * (-100)
                line_max_probs = line_probs.max(dim=-1)[0] + (1 - mask) * (-100)

                indices = torch.sort(edge_max_probs + line_max_probs, dim=-1, descending=True)[1]

                for ii in range(b):
                    keep = int((i + 1) / iterations * torch.sum(mask[ii, ...]))

                    assert torch.sum(mask[ii][indices[ii, :keep]]) == keep, "Error!!!"
                    mask[ii][indices[ii, :keep]] = 0

                mask = mask.reshape(b, 1, h, w)
                edge = edge * (1 - mask)
                line = line * (1 - mask)

            return edge, line
