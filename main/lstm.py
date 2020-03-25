"""
Implementation of the lstm model to create a vector representation of the question title and bodies from the ask ubuntu dataset.
Metrics map, mrrm p@1 and p@5 are also generated
"""


import torch
import util
# from util import util


def main():
    cuda = torch.cuda.is_available() and True
    num_epoch = 10
    batch_size = 2
    input_size = 200
    output_size = 120
    LR = 0.001
    dev_file = "../data/ask_ubuntu/dev.txt"
    test_file = "../data/ask_ubuntu/test.txt"
    train_file = "../data/ask_ubuntu/train_random.txt"
    corpus_file = "../data/ask_ubuntu/text_tokenized.txt"
    padding = "<padding>"
    embedding_path = "../data/ask_ubuntu/vectors_pruned.200.txt"

    print("-- lstm model --")
    print("embedding size:", output_size, "learning rate:", LR, "batch size:", batch_size, "num epoch:", num_epoch)


    # Represent each question as a word sequence (and not as a bag of words)
    data_loader = util.data_loader(corpus_file, cut_off=1, padding=padding)
    
    dev  = data_loader.read_annotations(dev_file, 20, 10)
    dev_data  = data_loader.create_eval_batches(dev)
    test = data_loader.read_annotations(test_file, 20, 10)
    test_data = data_loader.create_eval_batches(test)
    train_data = data_loader.read_annotations(train_file, 10, 2)
    
    # Utilize an exisiting vector representation of the words
    encoder = util.Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda)
    
    print("embeddings done")
    
    model = util.LSTM(input_size, output_size)
    if cuda:
        model = model.cuda()
        encoder = encoder.cuda()

    train_losses, dev_metrics, test_metrics \
        = util.train(encoder, model, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, util.LSTM_forward, True, cuda, LR)
    
    model = model.cpu()
    torch.save(model, "lstm.model")
    return train_losses, dev_metrics, test_metrics


if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    
    #  should graph these later
    print("training loss:", train_losses)
    print("metrics on the dev dataset:", dev_metrics)
    print("metrics on the test dataset:", test_metrics)
    
else:
    print("run the file directly to train the lstm model to create a semantic vector representation of the question title and associated body \
             from the ask ubuntu dataset and to generate metrics")   
