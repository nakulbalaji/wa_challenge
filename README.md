![answer](https://user-images.githubusercontent.com/30583886/221484456-e01d00d3-ff81-4080-818c-275027861564.png)

I used Python and OpenCV to do this perception challenge. I used opencv and numpy for my libraries. I separated the cones only using color ranges,
then using k-means clustering to get the contours properly. I had to change the kernel size for it to fit the situaton properly.
Then, getting the centers of these shapes, I made lines connecting each point. I also wanted to use the cv2.fitLine() function to 
get a general best fit line, but I didn't want to waste too much time trouble shooting that and used this approach instead.
Once I got the clusters (found using color), I separated them into 2 regions, the left and the right. This way, I could make two lines instead of my
original issue, which was all the lines criss-crossing each other and touching every cluster.
