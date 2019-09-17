/* Creator: Arto Viitanen
 * This macro is for fixing broken Aurox confocal images for the Imaris file converter & stitcher. To install the macro into ImageJ, go to 
 * Plugins > Macros > Install and then select this file.
 * 
 * The macro can be initiated by pressing key '1'. It will first ask for the directory with the image files, and then will ask for output directory.
 * When transferring the resulting imagefiles to Imaris file converter, you have to change the second to last field in "settings" in the tab "File Names 
 * with Delimiter" into "F (Split)". It should be set to "Z", but the files will NOT be converted correctly with that setting.
 * 
 * Notice: the macro splits the channels, and thus the file names are changed. Consequently, the metadata-file is inconsistent with the imagefile names.
 * 
 * Use split_channels and/or split_focal in the run("Bio-Formats", ...)-command to produce separate images for channels and/or z-layers, respectively
 * 
 * If not stacked on z-axis (separate image for each layer) use open_all_series as an argument in run("Bio-Formats" ...). If images of several midguts in 
 * same folder, do not use open_all_series.
*/
function save_as_tiff(input, output, file) {
	path = input + file;
	temp = split(file, ".");
	name = temp[0];
	run("Bio-Formats", "open=path color_mode=Default rois_import=[ROI manager] split_channels split_focal view=Hyperstack stack_order=XYCZT");
	number = nImages;
	for (x = 0; x < number; x++) {
		imageTitle=getTitle();
		strParts = split(imageTitle, ' ');
		if(strParts.length == 4){
			part1 = strParts[2];
			part2 = strParts[3];
			fullname = name+"_"+part1+"_"+part2;
		}else if(strParts.length == 3){
			part1 = strParts[2];
			fullname = name+"_"+part1;
		}else {
			fullname = name+"_"+x;
		}
		full_path = output + fullname;
		print(fullname);
		saveAs("Tiff", full_path);
		close();
	}
}
macro "Save folder as Tiffs[1]" {
	input = getDirectory("Choose input Directory");
	output = getDirectory("Choose output Directory");
	setBatchMode(true);
	list = getFileList(input);
	for (i=0; i < list.length; i++) {
		if(indexOf(list[i], ".tiff") >=0); {
			save_as_tiff(input, output, list[i]);
		}
	}
	setBatchMode(false);
}