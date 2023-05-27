function [dist, angle] = dictionary_distance(D1,D2)

% sweep through true and trained atoms and calculate inner product
D1D2 = D1'*D2;
dist = 1-abs(D1D2);
angle = acosd(D1D2);
% remove sign ambiguity in inner product
angle = (angle <= 90).*angle + (angle > 90).*(180-angle);
