function tilePos = gen_tile_pos(X, tileDim, mode, N)
% Returns indices of tiles to be extracted from image X
% tile size specified by 1x2 variable tileDim
% mode can be:
%   'fill' - extracts the maximum number of non-overlapping tiles
%   'random' - random tile positions (overlapping) without replicates
% N - number of tiles to extract; ignored for 'fill' mode

if strcmp(mode,'fill')
    rRange = 1:tileDim(1):size(X,1);
    tRange = 1:tileDim(2):size(X,2);
    tilePos(1,:) = repelem(rRange,length(tRange));
    tilePos(2,:) = repmat(tRange,1,length(rRange));
elseif strcmp(mode,'random')
    rRange = 1:(size(X,1)-tileDim(1)+1);
    tRange = 1:(size(X,2)-tileDim(2)+1);
    tilePos(1,:) = repelem(rRange,length(tRange));
    tilePos(2,:) = repmat(tRange,1,length(rRange));
    p = randperm(size(tilePos,2));
    if N <= length(p)
        tilePos = tilePos(:,p(1:N));
    else
        tilePos = tilePos(:,p);
    end
else
    disp('mode not recognised');
    tilePos = [];
    return;
end
