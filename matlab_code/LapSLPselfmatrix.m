function S = LapSLPselfmatrix(s) % single-layer Kress-split self-int Nyst matrix
% s = src seg, even # nodes. Barnett 10/18/13 based upon mpspack/@layerpot/S.m
N = numel(s.x);
d = repmat(s.x, [1 N]) - repmat(s.x.', [N 1]);    % C-# displacements mat
S = -log(abs(d)) + circulant(0.5*log(4*sin([0;s.t(1:end-1)]/2).^2)); % peri log
S(diagind(S)) = -log(s.sp);                       % diagonal limit
m = 1:N/2-1; Rjn = ifft([0 1./m 2/N 1./m(end:-1:1)])/2; % Kress Rj(N/2)/4pi
S = S/N + circulant(Rjn); % includes SLP prefac 1/2pi. Kress peri log matrix L
S = S .* repmat(s.sp.',[N 1]);  % include speed factors (not 2pi/N weights)

end

function A = circulant(x)
% function A = circulant(x)
%
% return square circulant matrix with first row x
% barnett 2/5/08
x = x(:);
A = toeplitz([x(1); x(end:-1:2)], x);
end

function i = diagind(A)
% function i = diagind(A)
%
% return diagonal indices of a square matrix, useful for changing a diagonal
% in O(N) effort, rather than O(N^2) if add a matrix to A using matlab diag()
%
% barnett 2/6/08
N = size(A,1);
if size(A,2)~=N
  disp('input must be square!');
end
i = sub2ind(size(A), 1:N, 1:N);
end