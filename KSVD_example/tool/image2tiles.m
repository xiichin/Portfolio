function Y = image2tiles(X, tileDim, tilePos)

% extracts tiles from image X, vectorise them and return them as Y
% tile size specified by tileDim (1x2),
% tile positions specified by tilePos (2xN).
% where N is the number of vectors in the output matrix

N = prod(tileDim);  % output vector length
M = size(tilePos,2);

if size(tilePos,1) ~= 2
    disp('must specify tile positions as 2xN matrix');
    return;
end

Y = zeros(N,M);
colRange = 0:(tileDim(1)-1);
rowRange = 0:(tileDim(2)-1);
for ii = 1:M
    Y(:,ii) = reshape(X(tilePos(1,ii)+colRange,tilePos(2,ii)+rowRange), N, 1);
end
