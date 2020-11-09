python train_age.py \
	--audience_dataset '/home/rishi/Projects/Matroid/Rishi_Challenge/Data/adience_data_preprocessing/faces/faces_dataset.h5' \
	--vgg_weight '/home/rishi/Projects/Matroid/Rishi_Challenge/Data/vgg_data_preprocessing/pkl_weights' \
	--folder_to_test 1 \
	--dropout_keep_prob 0.5 \
	--weight_decay 1e-2 \
	--learning_rate 1e-2 \
	--batch_size 64 \
	--num_epochs 50 \
	--evaluate_every 2 \
	--moving_average True \

	
