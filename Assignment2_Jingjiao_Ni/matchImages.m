function [match_fracs] = matchImages(imdb_path, seq_cutoff, verify_geom)
tic
% Compares each image in imdb with all images upto 'seq_cutoff' time before it.
% The degree of match between two images is measured by the fraction of
% words that they have in common and they are considered matched if this 
% is not less than 'match_thresh'.
if nargin<3
    verify_geom=1; % set this to 0 to turn off geometric verification step
end
if nargin<2
    seq_cutoff=10; % no. of images before the current image that are not matched
end
if nargin<1
    imdb_path='loop_closure_imdb.mat';
end
load(imdb_path);
no_of_images=numel(imdb.images.id);
if verify_geom
    disp('Geometric verification is enabled');
end

% Matrix for computing loop closure detection 
match_fracs=double(zeros(no_of_images));

% %%%%%%%  pseudo-code outline for matching %%%%%%%%
%
% for every pair of images that are at least seq_cutoff apart,
% call "matchWords" function to obtain number of matched words.
%
% If verify_geom variable is 0, divide number of matched words by the 
% number of words present in the image pair to compute the degree of match 
% between the image pair. Populate match_fracs matrix with the degree of match
%
% If verify_geom variable is 1, call "geometricVerification" function,
% which outputs number of inliers. Divide number of inliers by the 
% number of words present in the image pair to compute the degree of match 
% between the image pair. Populate match_fracs matrix with the degree of match
%
% Call supplied function "convertToBinary" to decide a suitable threshold
% for match_fractions. Match your binary matrix output (O) with ground truth 
% binary matrix (G) by computing Jaccard score: sum(sum(and(O,G)))/ sum(sum(or(O,G))) 

for i=seq_cutoff+1:no_of_images
    for j=1:i-seq_cutoff       
        matches = matchWords(imdb.images.words{i},imdb.images.words{j}); 
        if size(matches,2)~=0
            if verify_geom     
                [inliers,~] = geometricVerification(imdb.images.frames{i},imdb.images.frames{j},matches);
                match_fracs(i,j)=size(inliers,2)/max(size(imdb.images.words{i},2),size(imdb.images.words{j},2));
            else
                match_fracs(i,j)=size(matches,2)/max(size(imdb.images.words{i},2),size(imdb.images.words{j},2));
            end    
        end
    end
end
toc
end