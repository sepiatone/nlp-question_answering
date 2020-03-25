"""
Implementation of the cnn model to create a vector representation of the question title and bodies from the ask ubuntu dataset.
Metrics map, mrrm p@1 and p@5 are also generated
"""

 
import torch
import util
# from util import util


def main():
    cuda = torch.cuda.is_available() and True
    embedding_size = 200
    convolution_size = 3
    LR= 0.01
    CNN_size = 667
    batch_size = 5
    num_epoch = 1 # 10
    
    print("-- cnn model details --")
    print("embedding Size:", CNN_size, "convolution size:", convolution_size, "learning rate:", LR, "batch size:", batch_size, "num epochs:", num_epoch)

    padding = "<padding>"
    train_file = "../data/ask_ubuntu/train_random.txt"
    dev_file = "../data/ask_ubuntu/dev.txt"
    test_file = "../data/ask_ubuntu/test.txt"
    corpus_file = "../data/ask_ubuntu/text_tokenized.txt"
    embedding_path = "../data/ask_ubuntu/vectors_pruned.200.txt"

    data_loader = util.data_loader(corpus_file, cut_off = 2, padding=padding)

    encoder = util.Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda)

    print("loaded encoder")
    CNN = util.CNN(embedding_size, CNN_size, convolution_size)
    
    if cuda:
        encoder = encoder.cuda()
        CNN = CNN.cuda()

    print("loading annotations...")
    dev  = data_loader.read_annotations(dev_file, 20, 10)
    dev_data  = data_loader.create_eval_batches(dev)
    test = data_loader.read_annotations(test_file, 20, 10)
    test_data = data_loader.create_eval_batches(test)
    train_data = data_loader.read_annotations(train_file)
    print("loaded annotations")

    train_losses, dev_metrics, test_metrics = \
        util.train(encoder, CNN, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, util.CNN_forward, True, cuda, LR=LR)
    
    CNN = CNN.cpu()
    torch.save(CNN, "cnn.model")
    
    return train_losses, dev_metrics, test_metrics


if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    
    #  should graph these later
    print("training loss:", train_losses)
    print("metrics on the dev dataset:", dev_metrics)
    print("metrics on the test dataset:", test_metrics)
    
else:
    print("run the file directly to train the cnn model to create a semantic vector representation of the question title and associated body \
             from the ask ubuntu dataset and to generate metrics")