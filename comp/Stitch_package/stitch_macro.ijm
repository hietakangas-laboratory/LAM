paths = getArgument()
print(paths);

listy = split(paths, " ")

companion_ome = listy[1]
print(companion_ome)
fused_tiff = listy[0]
print(fused_tiff)

run("Bio-Formats Macro Extensions");
Ext.openImagePlus(companion_ome)

getDimensions(width, height, channels, slices, frames)
ch_no = channels 
print(ch_no)
run("Close All")

print(fused_tiff)
open(fused_tiff, "virtual")
iddy2 = getTitle()
print(iddy2)
selectWindow(iddy2)

if (is("composite")){
x = true;
} else {
	x = false;
}

print(x);

if (x && ch_no == 4){
print("processing composite");
print("performing 4ch macro");
//run("Channels Tool...");
Stack.setDisplayMode("color");
Stack.setChannel(1);
run("Red");
Stack.setChannel(2);
run("Green");
Stack.setChannel(3);
run("Magenta");
Stack.setChannel(4);
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
Stack.setChannel(1);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(2);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(3);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(4);
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
print("done ptI");
} else if (x && ch_no == 3){
print("processing composite");
print("performing 3ch macro");
//run("Channels Tool...");
Stack.setDisplayMode("color");
Stack.setChannel(1);
run("Red");
Stack.setChannel(2);
run("Green");
Stack.setChannel(3);
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
Stack.setChannel(1);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(2);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(3);
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
} else if (x && ch_no == 2){
print("processing composite");
print("performing 2ch macro");
//run("Channels Tool...");
Stack.setDisplayMode("color");
Stack.setChannel(1);
run("Green");
Stack.setChannel(2);
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
Stack.setChannel(1);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(2);
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
} else if (x && ch_no == 1) {
print("processing composite");
print("performing 1ch macro");
//run("Channels Tool...");
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
} else if (x == 0 && ch_no == 4){
print("performing 4ch macro");
s = 1;
t = nSlices();
i = t/4;
startSlice1 = s;
endSlice1 = i;
startSlice2 = s+i;
endSlice2 = 2*i;
startSlice3 = s+(2*i);
endSlice3 = 3*i;
startSlice4 = s+(3*i);
endSlice4 = 4*i;
   selectWindow(iddy2);
  run("Make Substack...", "  slices="+startSlice1+"-"+endSlice1+"");
   selectWindow(iddy2);
   run("Make Substack...", "  slices="+startSlice2+"-"+endSlice2+"");
    selectWindow(iddy2);
  run("Make Substack...", "  slices="+startSlice3+"-"+endSlice3+"");
   selectWindow(iddy2);
   run("Make Substack...", "  slices="+startSlice4+"-"+endSlice4+"");
run("Merge Channels...", "c1=[Substack ("+startSlice1+"-"+endSlice1+")] c2=[Substack ("+startSlice2+"-"+endSlice2+")] c3=[Substack ("+startSlice3+"-"+endSlice3+")] c4=[Substack ("+startSlice4+"-"+endSlice4+")] create");
compositetitle=getTitle();
selectWindow(compositetitle);
//run("Channels Tool...");
Stack.setDisplayMode("color");
Stack.setChannel(1);
run("Red");
Stack.setChannel(2);
run("Green");
Stack.setChannel(3);
run("Magenta");
Stack.setChannel(4);
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
Stack.setChannel(1);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(2);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(3);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(4);
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "channels=4 slices="+i+" frames=1 unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
} else if (x == 0 && ch_no == 3){
print("performing 3ch macro");
s = 1;
t = nSlices();
i = t/3;
startSlice1 = s;
endSlice1 = i;
startSlice2 = s+i;
endSlice2 = 2*i;
startSlice3 = s+(2*i);
endSlice3 = 3*i;
startSlice4 = s+(3*i);
endSlice4 = 4*i;
  selectWindow(iddy2);
  run("Make Substack...", "  slices="+startSlice1+"-"+endSlice1+"");
   selectWindow(iddy2);
   run("Make Substack...", "  slices="+startSlice2+"-"+endSlice2+"");
    selectWindow(iddy2);
  run("Make Substack...", "  slices="+startSlice3+"-"+endSlice3+"");
run("Merge Channels...", "c1=[Substack ("+startSlice1+"-"+endSlice1+")] c2=[Substack ("+startSlice2+"-"+endSlice2+")] c3=[Substack ("+startSlice3+"-"+endSlice3+")] create");
compositetitle=getTitle();
selectWindow(compositetitle);
//run("Channels Tool...");
Stack.setDisplayMode("color");
Stack.setChannel(1);
run("Red");
Stack.setChannel(2);
run("Green");
Stack.setChannel(3);
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
Stack.setChannel(1);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(2);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(3);
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "channels=3 slices="+i+" frames=1 unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
} else if (x == 0 && ch_no == 2){
print("performing 2ch macro");
s = 1;
t = nSlices();
i = t/2;
startSlice1 = s;
endSlice1 = i;
startSlice2 = s+i;
endSlice2 = 2*i;
startSlice3 = s+(2*i);
endSlice3 = 3*i;
startSlice4 = s+(3*i);
endSlice4 = 4*i;
  selectWindow(iddy2);
  run("Make Substack...", "  slices="+startSlice1+"-"+endSlice1+"");
   selectWindow(iddy2);
   run("Make Substack...", "  slices="+startSlice2+"-"+endSlice2+"");
run("Merge Channels...", "c1=[Substack ("+startSlice1+"-"+endSlice1+")] c2=[Substack ("+startSlice2+"-"+endSlice2+")] create");
compositetitle=getTitle();
selectWindow(compositetitle);
//run("Channels Tool...");
Stack.setDisplayMode("color");
Stack.setChannel(1);
run("Green");
Stack.setChannel(2);
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
Stack.setChannel(1);
run("Enhance Contrast", "saturated=0.35");
Stack.setChannel(2);
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "channels=2 slices="+i+" frames=1 unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
} else if (x == 0 && ch_no == 1){
print("performing 1ch macro");
s = 1;
t = nSlices();
i = t;
startSlice1 = s;
endSlice1 = i;
startSlice2 = s+i;
endSlice2 = 2*i;
startSlice3 = s+(2*i);
endSlice3 = 3*i;
startSlice4 = s+(3*i);
endSlice4 = 4*i;
  selectWindow(iddy2);
  run("Make Substack...", "  slices="+startSlice1+"-"+endSlice1+"");
run("Merge Channels...", "c1=[Substack ("+startSlice1+"-"+endSlice1+")] create");
compositetitle=getTitle();
selectWindow(compositetitle);
//run("Channels Tool...");
run("Cyan");
//run("Brightness/Contrast...");
//run("Channels Tool...");
run("Enhance Contrast", "saturated=0.35");
run("Rotate 90 Degrees Right");
run("Properties...", "channels=1 slices="+i+" frames=1 unit=um pixel_width=0.3250000 pixel_height=0.3250000 voxel_depth=3.4000000");
} else {
print("Macro could not be performed on " + fused_tiff)
}
print("macro process completed on" + fused_tiff);