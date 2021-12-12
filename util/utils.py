import os
import numpy as np
import imageio
import time,datetime
import matplotlib.pyplot as plt
from config import cfg

def imshow(img, title=""):
    """ Show image as grayscale. 
    imshow(np.linalg.norm(coilImages, axis=0))
    """
    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.show()


def lr_plot(lr_, save_dir):
    # iters = range(len(psnr_[psnr_type]))
    iters = np.linspace(1, len(lr_), len(lr_))
    min_idx = np.argmin(lr_)
    
    fig = plt.figure()
    plt.title('Restoration on {}'.format('ComplexMRI'))
    plt.plot(iters, lr_, 'g', label='epoch_psnr')
    plt.plot(min_idx+1, lr_[min_idx], marker='.', color='k')
    plt.annotate('min:{:.6f}'.format(lr_[min_idx]),
                                xytext=(min_idx-5,lr_[min_idx]),
                                xy=(min_idx,lr_[min_idx]),
                                textcoords='data'
                                )
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.legend(loc="best")
    a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    plt.savefig(os.path.join(save_dir,'lr{}.png'.format(a)))
    plt.close(fig)


def plot_loss(dst_dict, dst_type, save_dir, args=cfg):
    assert type(dst_type) is str

    if dst_type.lower().find('client')>=0:
        for idx, client_name in enumerate(args.DATASET.CLIENTS):
            client_dir = os.path.join(args.LOGDIR, client_name)
            os.makedirs(client_dir, exist_ok=True)

            iters = np.linspace(1, args.TRAIN.EPOCHS, args.TRAIN.EPOCHS)
            
            fig = plt.figure()
            plt.title('Restoration on {}'.format(args.FL.MODEL_NAME))

            # train loss
            input_list_train = dst_dict[idx]['loss_train_epoch']
            plt.plot(iters, input_list_train, 'firebrick', label='train loss')
            
            # val loss
            input_list_val = dst_dict[idx]['loss_val_epoch']
            plt.plot(iters, input_list_val, 'darkorange', label='val loss')

            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            ax = plt.gca()
            ax.yaxis.get_major_formatter().set_powerlimits((0,2))  # 将坐标轴的base number设置为2位
            plt.legend(loc="best")
            plt.savefig(os.path.join(client_dir,'Loss_{}.png'.format(client_name)))
            plt.close(fig)

    elif dst_type.lower().find('train')>=0 or dst_type.lower().find('val')>=0:
        iters = np.linspace(1, args.TRAIN.EPOCHS, args.TRAIN.EPOCHS)
        colors = ['firebrick', 'darkorange', 'forestgreen', 'darkcyan', 'royalblue', 'darkorchid']
        input_list_train = []

        fig = plt.figure()
        plt.title('Reconstruction on {}'.format(args.FL.MODEL_NAME))
        
        for idx, client_name in enumerate(args.DATASET.CLIENTS):

            if dst_type.lower().find('train')>=0:
                # train loss
                input_list_train = dst_dict[idx]['loss_train_epoch']
                plt.plot(iters, input_list_train, label='train_{}'.format(client_name), color=colors[idx])
            
            if dst_type.lower().find('val')>=0:
                linestyle_val = '--' if input_list_train != [] else '-'

                # val loss
                input_list_val = dst_dict[idx]['loss_val_epoch']
                plt.plot(iters, input_list_val, label='val_{}'.format(client_name), color=colors[idx], linestyle=linestyle_val)

        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((0,2))  # 将坐标轴的base number设置为2位
        plt.legend(loc="best")
        plt.savefig(os.path.join(args.LOGDIR,'Loss_{}.png').format(dst_type))
        plt.close(fig)

    else:
        raise ValueError('Wrong dst_type: {}!'.format(dst_type))


def any_plot(dst_dict, dst_type, save_dir):
    assert type(dst_type) is str
    if dst_type.lower() == 'loss':
        # train loss
        input_list_train = dst_dict['loss_train_count']
        iters = np.linspace(1, 50, len(input_list_train))  # from 1 to ...
        idx_min_train = np.argmin(input_list_train)
        fig = plt.figure()
        plt.title('Restoration on {}'.format('ComplexMRI'))
        
        
        plt.plot(iters, input_list_train, 'g', label='train loss')
        # plt.plot(idx_min_train+1, input_list_train[idx_min_train], marker='v', color='k')  # 标记点
        # plt.annotate('min:{:.6f}'.format(input_list_train[idx_min_train]),  # 标记数据
        #                             xytext=(idx_min_train-5,input_list_train[idx_min_train]),
        #                             xy=(idx_min_train,input_list_train[idx_min_train]),
        #                             textcoords='data'
        #                             )
        
        # val loss
        input_list_val = dst_dict['loss_val_count']
        idx_min_val = np.argmin(input_list_val)
        plt.plot(iters, input_list_val, 'r', label='val loss')
        # plt.plot(idx_min_val+1, input_list_val[idx_min_val], marker='v', color='k')  # 标记点
        # plt.annotate('min:{:.6f}'.format(input_list_val[idx_min_val]),  # 标记数据
        #                             xytext=(idx_min_val-5,input_list_val[idx_min_val]),
        #                             xy=(idx_min_val,input_list_val[idx_min_val]),
        #                             textcoords='data'
        #                             )
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel(dst_type)
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((0,2))
        plt.legend(loc="best")
        a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        plt.savefig(os.path.join(save_dir,'Loss_{}.png'.format(a)))
        plt.close(fig)
    
    else:
        print(dst_type.lower())
        if dst_type.lower() == 'psnr':
            input_list = dst_dict['psnr_epoch']
        elif dst_type.lower() == 'ssim':
            input_list = dst_dict['ssim_epoch']
        elif dst_type.lower() == 'lr' or dst_type.lower().find('learning rate')>=0:
            input_list = dst_dict['lr_epoch']
        else: raise ValueError('Invalid Criterion: {}'.format(dst_type))

        iters = np.linspace(1, len(input_list), len(input_list))
        max_idx = np.argmax(input_list)
        
        fig = plt.figure()
        plt.title('Restoration on {}'.format(args.name))
        plt.plot(iters, input_list, 'g', label='{}_epoch'.format(dst_type))
        plt.plot(max_idx+1, input_list[max_idx], marker='v', color='k')
        plt.annotate('max:{:.3f}'.format(input_list[max_idx]),
                                    xytext=(max_idx,input_list[max_idx]),
                                    xy=(max_idx,input_list[max_idx]),
                                    textcoords='data'
                                    )
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel(dst_type)
        plt.legend(loc="best")
        a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        plt.savefig(os.path.join(save_dir,'{}_{}.png'.format(dst_type,a)))
        plt.close(fig)

# psnr_plot(psnr_, 'epoch', save_dir=experiment_dir)
def psnr_plot(psnr_, psnr_type, save_dir):
    # iters = range(len(psnr_[psnr_type]))
    iters = np.linspace(1, len(psnr_[psnr_type]), len(psnr_[psnr_type]))
    max_idx = np.argmax(psnr_[psnr_type])
    
    fig = plt.figure()
    plt.title('Restoration on {}'.format('ComplexMRI'))
    plt.plot(iters, psnr_[psnr_type], 'g', label='epoch_psnr')
    plt.plot(max_idx+1, psnr_[psnr_type][max_idx], marker='v', color='k')
    plt.annotate('max:{:.3f}'.format(psnr_[psnr_type][max_idx]),
                                # xytext=(max_idx-5,psnr_[psnr_type][max_idx]),
                                xytext=(max_idx,psnr_[psnr_type][max_idx]),
                                xy=(max_idx,psnr_[psnr_type][max_idx]),
                                textcoords='data'
                                )
    plt.grid(True)
    plt.xlabel(psnr_type)
    plt.ylabel('PSNR')
    plt.legend(loc="best")
    a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    plt.savefig(os.path.join(save_dir,'PSNR{}.png'.format(a)))
    plt.close(fig)

if __name__ == '__main__':

    iters = np.linspace(1, 50, 50)
    
    fig = plt.figure()
    plt.title('Restoration on debug')

    colors = ['firebrick', 'darkorange', 'forestgreen', 'darkcyan', 'royalblue', 'darkorchid']

    # train loss
    input_list_train = np.linspace(5, 54, 50)
    plt.plot(iters, input_list_train, 'forestgreen', label='train loss')
    
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,2))  # 将坐标轴的base number设置为2位
    plt.legend(loc="best")
    plt.savefig(os.path.join('Loss_debug.png'))
    plt.close(fig)

    print('done')