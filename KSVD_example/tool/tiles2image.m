function X = tiles2image(Y, tileDim, tilePos)

if size(Y,1) ~= prod(tileDim)
    disp('dimensions must agree, abort');
    return;
end

X = zeros(max(tilePos(1,:))+tileDim(1)-1,max(tilePos(2,:))+tileDim(2)-1);
colRange = 0:(tileDim(1)-1);
rowRange = 0:(tileDim(2)-1);
for ii = 1:size(Y,2)
    X(tilePos(1,ii)+colRange,tilePos(2,ii)+rowRange) = reshape(Y(:,ii),tileDim(1),tileDim(2));
end
