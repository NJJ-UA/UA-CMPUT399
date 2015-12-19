function [imdb] = constructIMDB(dir_name, count, no_of_words)
% reads images from the given directory and constructs a database
tic
% default parameters
if nargin<3
    no_of_words=10000;
end
if nargin<2
    count=1063;
end
if nargin<1
    dir_name='Lip6OutdoorDataSet/Images';
end

if no_of_words==100000
    use_100k_flag=1;
    disp('Use the 100k vocab');
else
    use_100k_flag=0;
    disp('Does not use the 100k vocab');
end

% Step 1: Initialize imdb structure
images.name=cell(1, count);
images.id=1:count;
images.frames=cell(1, count);
images.words=cell(1, count);
images.descrs=cell(1, count);
imdb.images=images;
imdb.dir=dir_name;
imdb.featureOpts={'method','dog','affineAdaptation',true,'orientation',false};
imdb.numWords=no_of_words;
imdb.sqrtHistograms=0;

% a variable where you will save the imdb database later
save_template='loop_closure_imdb'; 

% Step 2: Read images and extract features
% Here you will be reading images, calling "getFeatures" function and populating
% imdb.images.name, imdb.images.frames and imdb.images.descrs cell arrays

image_list=dir(strcat(dir_name,'/*.ppm'));

for i=1:count
    if mod(i, 100)==0
        fprintf('Processed %d images\n', i);
    end
    filename=image_list(i).name;
     
    imdb.images.name{i}=filename;

    img=imread(strcat(dir_name, '/',filename));
    if size(img, 3) == 3 % image is RGB
        img=rgb2gray(img);
    end
    
    [frames, descrs, ~] = getFeatures(img);
    imdb.images.frames{i}=frames;
    imdb.images.descrs{i}=descrs;
end


% Step 3: Construct a vocabulary of words from the combined features of all the images
% using K-Means clustering. Call "vl_kmeans" function with combined features and
% imdb.numWords to assign vocabulary to imdb.vocab


D=cell2mat(imdb.images.descrs);

if use_100k_flag
    imdb_100k=load('oxbuild_lite_imdb_100k_ellipse_dog.mat');
    imdb.vocab=imdb_100k.vocab;
   % imdb.kdtree=imdb_100k.kdtree;
    imdb.kdtree=vl_kdtreebuild(imdb.vocab);
else
    [imdb.vocab, ~] = vl_kmeans(D,imdb.numWords,'Algorithm','ANN');
    
% Step 4: Construct a KD tree from this vocabulary and assign to imdb.kdtree

    imdb.kdtree=vl_kdtreebuild(imdb.vocab);

end


% Step 5: Find the words present in each image through NN search of the
% vocabulary to populate imdb.images.words cell array. Use "vl_kdtreequery" 
% function for the NN search.

[index,~]=vl_kdtreequery(imdb.kdtree,imdb.vocab,D,'NUMNEIGHBORS',1);%'MaxNumComparisons',10000);

start=1;
for i=1:count
    n=size(imdb.images.descrs{i},2);
    ending=start+n-1;
    imdb.images.words{i}=index(:,start:ending);
    start=ending+1;
end
% Step 6: Compute indexes and idf weights by calling "loadIndex" function

% Step 7: Save imdb in a .mat file
save(strcat(save_template, '.mat'), 'imdb');
toc