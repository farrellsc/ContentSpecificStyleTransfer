wget -O ../models/saved_models.zip https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1
mkdir ../models/JohnsonNet/
unzip ../models/saved_models.zip -d ../models/JohnsonNet/
mv ../models/JohnsonNet/saved_models/* ../models/JohnsonNet/
rmdir ../models/JohnsonNet/saved_models
rm ../models/saved_models.zip