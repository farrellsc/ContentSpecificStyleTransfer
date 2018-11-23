from PIL import Image
import os
import sys


def resize_image(base_path, output_path):
	size = 256
	cnt = 0

	for subdir in os.listdir(base_path):
		for picname in os.listdir(base_path + subdir):
			infile = base_path + subdir + "/" + picname
			outfile = output_path + picname
			im = Image.open(infile)
			if im.size[0] < size or im.size[1] < size: continue
			if im.size[0] <= im.size[1]:
				ratio = im.size[0] / 256
				im = im.resize((256, round(im.size[1]/ratio)))
			else:
				ratio = im.size[1] / 256
				im = im.resize((round(im.size[0]/ratio), 256))
			im.save(outfile, "JPEG")
			cnt+=1
			if cnt == 5000: break
			if cnt % 100 == 0: print(cnt)
		if cnt == 5000: break


if __name__ == "__main__":
	base_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/wiki_crop/"
	output_path = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/portrait/"
	resize_image(base_path, output_path)
