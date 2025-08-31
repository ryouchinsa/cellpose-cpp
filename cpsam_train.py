import argparse
from cellpose import io, models, train
io.logger_setup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",type=str,default="show",required=True,help="training images directory")
    parser.add_argument("--n_epochs",type=int,default=100,required=False,help="training epochs")
    parser.add_argument("--device",type=str,default="cpu",required=False,help="cpu or cuda:0")
    args = parser.parse_args()
    if args.device == "cpu":
        gpu = False
    else:
        gpu = True
    output = io.load_train_test_data(args.train_dir, mask_filter="_masks", )
    images, labels, image_names, test_images, test_labels, image_names_test = output
    model = models.CellposeModel(gpu=gpu, use_bfloat16=False)
    model_path, train_losses, test_losses = train.train_seg(model.net,
                                train_data=images, train_labels=labels,
                                test_data=test_images, test_labels=test_labels,
                                weight_decay=0.1, learning_rate=1e-5,
                                n_epochs=args.n_epochs, model_name="my_new_model")