import os, time

from options import test_options
from dataloader import data_loader
from model import create_model
# from util import visualizer

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    # visualizer = visualizer.Visualizer(opt)

    # 构建用于保存测试结果的目录路径，其中包括结果的根目录\测试的轮次\掩码率并打印
    mask_name = os.path.basename(opt.mask_file)  # 获取掩码路径最后一个文件名也就是掩码率
    save_dir = os.path.join(opt.results_dir, opt.which_iter, '{}'.format(mask_name))
    print('creating save directory', save_dir)
    opt.results_dir = save_dir
    start_time = time.time()

    """
    for i, data in enumerate(islice(dataset, opt.how_many)):
        model.set_input(data)
        model.test()
    """
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()

    total_time = time.time() - start_time
    print('the total evaluation time %f' % (total_time))