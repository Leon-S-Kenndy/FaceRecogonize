file_path='E:\ѧϰ\�����\����Ӫ\face\test\'
img_path_list = dir(strcat(file_path,'*.pgm'))
img_num = length(img_path_list);
 names=0;
for i=1:img_num
    mImageSrc= imread(strcat(file_path,img_path_list(i).name));
    x=size(mImageSrc,3);
    mImage2detect=[]
    mImage2detect=uint8(mImage2detect)
    if(size(mImageSrc,3) == 1) 
        mImage2detect(:,:,1) = mImageSrc;
        mImage2detect(:,:,2) = mImageSrc;
        mImage2detect(:,:,3) = mImageSrc;
    else
        mImage2detect = mImageSrc;
    end

    FaceDetector               = buildDetector(); 
    [bbox,bbimg,faces,bbfaces] = detectFaceParts(FaceDetector,mImage2detect,2);

    leng=size(bbox,1);
    %imshow(bbimg);
    for j= 1: size(faces,1)
        %figure
        xx=cell2mat(faces(j,1));
        %imshow(xx);
        names=names+1;
        imwrite(xx,strcat('E:\ѧϰ\�����\����Ӫ\face\test_face\',num2str(names),'.jpg'))
    end
end


