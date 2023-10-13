import logging
import time
import torch
import torch_npu
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from sklearn.cluster import DBSCAN
from utils.re_ranking import re_ranking


def generate_pseudo_labels(cluster_id, num, train_loader):
            labels = []
            outliers = 0
            for i, (fname, id) in enumerate(zip(sorted(train_loader.dataset.dataset), cluster_id)):
                if id!=-1:
                    labels.append(id)
                else:
                    labels.append(num+outliers)
                    outliers += 1
            return torch.Tensor(labels).long()


def extract_features(model, train_loader):
        model.eval()
        device = torch.device("npu")
        
        
        images_feats = torch.empty((0,512)).to(device)
        texts_feats = torch.empty((0,512)).to(device)
        
        with torch.no_grad():
            for n_iter, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                images = batch['images']
                caption_ids = batch['caption_ids']
                
                image_feats, text_feats = model.base_model(images, caption_ids)
                
                i_feats = image_feats[:, 0, :].float()
                # i_feats = image_feats.float() # for CLIP ResNet visual model
                t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
                
                images_feats = torch.cat((images_feats, i_feats), dim=0)
                texts_feats = torch.cat((texts_feats, i_feats), dim=0)
                
            print('=================================')

        return images_feats, texts_feats
    

def extract_features(model, train_loader):
        model.eval()
        device = torch.device("npu")
        
        
        images_feats = torch.empty((0,512)).to(device)
        texts_feats = torch.empty((0,512)).to(device)
        
        with torch.no_grad():
            for n_iter, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                images = batch['images']
                caption_ids = batch['caption_ids']
                
                image_feats, text_feats = model.base_model(images, caption_ids)
                
                i_feats = image_feats[:, 0, :].float()
                # i_feats = image_feats.float() # for CLIP ResNet visual model
                t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
                
                images_feats = torch.cat((images_feats, i_feats), dim=0)
                texts_feats = torch.cat((texts_feats, i_feats), dim=0)
                
            print('=================================')

        return images_feats, texts_feats


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    # device = "cuda"
    device = torch.device("npu")
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        
        images_features, texts_features = extract_features(model, train_loader)
        
        rerank_dist = re_ranking(images_features,  k1=args.k1, k2=args.k2)
        
        if (epoch==1):
            eps = args.eps
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids, train_loader)
        
        args.num_clusters = num_ids
        
    
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # ret = model(batch,pseudo_labels)
            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            # meters['cluster_loss'].update(ret.get('cluster_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                # torch.cuda.empty_cache()
                torch_npu.npu.empty_cache()
                
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                    
                logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")
    
    # if get_rank() == 0:
    #     logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
