"""
implementation of transfer learning with adversarial domain adaptation - i.e., the cnn and lstm models trained on the ask ubuntu corpus
are evaluated on the stackexchange android dataset.
"""


import torch
import util


def main():
    use_lstm = True
    cuda = torch.cuda.is_available() and True
    embedding_size = 300
    convolution_size = 3
    output_size = 200
    cnn_output_size = 600
    batch_size = 2
    num_epoch = 10
    classifier_hidden_size = 20
    lr = 0.01

    padding = "<padding>"
    embedding_path = "../data/glove.txt"
    corpus_file = "../data/ask_ubuntu/text_tokenized.txt"
    corpus_2 = "../data/stackexchange_android/corpus.txt"
    train_file = "../data/ask_ubuntu/train_random.txt"    
    pos_dev = "../data/stackexchange_android/dev.pos.txt"
    neg_dev = "../data/stackexchange_android/dev.neg.txt"
    pos_test = "../data/stackexchange_android/test.pos.txt"
    neg_test = "../data/stackexchange_android/test.neg.txt"
    
    if use_lstm:
        print("-- lstm model --")
        print("embedding size:", embedding_size, "learning rate:", lr, "batch size:", batch_size, "num epoch:", num_epoch)
    else:
        print("-- cnn model --")
        print("embedding size:", embedding_size, "learning rate:", lr, "batch size:", batch_size, "num epoch:", num_epoch,\
               "convolution size:", convolution_size, "cnn output size:", cnn_output_size)        

    data_loader = util.data_loader(corpus_file, cut_off=1, padding=padding)
    data_loader.read_new_corpus(corpus_2)

    dev_annotations = util.read_annotations_2(pos_dev, neg_dev, -1, -1)
    test_annotations = util.read_annotations_2(pos_test, neg_test, -1, -1)
    dev_data = data_loader.create_eval_batches(dev_annotations, first_corpus=False)
    test_data = data_loader.create_eval_batches(test_annotations, first_corpus=False)

    train_data = data_loader.read_annotations(train_file)

    if use_lstm:
        print("-- lstm model --")
        print("embedding size:", embedding_size, "learning rate:", lr, "batch size:", batch_size, "num epoch:", num_epoch)
        
        model = util.LSTM(embedding_size, output_size)
        forward = util.LSTM_forward
        classifier = util.Classifier(output_size, classifier_hidden_size)
    else:
        print("-- cnn model --")
        print("embedding size:", embedding_size, "learning rate:", lr, "batch size:", batch_size, "num epoch:", num_epoch,\
               "convolution size:", convolution_size, "cnn output size:", cnn_output_size)
        
        model = util.CNN(embedding_size, cnn_output_size, convolution_size)
        forward = util.CNN_forward
        classifier = util.Classifier(cnn_output_size, classifier_hidden_size)

    encoder = util.Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda, embedding_size)

    print("encoder loaded")
    
    if cuda:
        model = model.cuda()
        encoder = encoder.cuda()
        classifier = classifier.cuda()

    return util.advisarial_trainer(encoder, model, classifier, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, cuda, LR=0.0001, L=False)


if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    
    #  should graph these later
    print("training loss:", train_losses)
    print("metrics on the dev dataset:", dev_metrics)
    print("metrics on the test dataset:", test_metrics)
    
else:
    print("run the file directly to evaluate the model on the stackexchange android corpus and to generate metrics")
