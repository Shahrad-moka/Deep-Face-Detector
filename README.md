# Deep-Face-Detector
This network is using VGG16 for transfer-learning. I have used 90 self-collected images and augmented the data to create about 4700 images. there are two loss functions which will be summed up as the total loss. in the end, real-time face detection and bounding-box-generation are implemented using open-cv. 
Organize the directories as follows:
mani project folder.list_dir: [Data, Aug_data]
Data.list_fir: [Images, Labels, Test, Train, Val]
Only for Test, Train, and Val, include [Images, Labels] idntically'
Now let`s go back to the main project folder and navigate to Aug_data
Aug_data.list_dir: [Test, Train, Val]
Repeat line 6
