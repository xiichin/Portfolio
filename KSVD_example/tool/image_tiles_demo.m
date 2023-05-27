% example how to use tiles2image and image2tiles

% X = double(imread('cameraman.tif'));    % read in image
X = double(imread('1 no.png'));

% set parameters
tileDim = [8 8];
mode = 'fill';
nTiles = 0;

% generate tile positions - needed to reassemble later
tilePos = gen_tile_pos(X, tileDim, mode, nTiles);
% extract tiles into column vectors and stack to give matrix Y
Y = image2tiles(X, tileDim, tilePos);
% reassemble image from tiles
X2 = tiles2image(Y, tileDim, tilePos);

% check result
error = norm(X-X2,'fro')
