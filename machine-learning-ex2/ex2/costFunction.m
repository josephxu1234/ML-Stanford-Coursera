function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%summation for calculating J (cost function)
sumJ = 0;
%summation for calculating G (gradient descent)
sumG = zeros(size(theta));

for i = 1:m,
	sumJ += y(i) * log(hyp (theta, X(i, :)')) + (1 - y(i)) * log(1 - hyp(theta, X(i, :)'));
	sumG += (y(i) - hyp(theta, X(i, :)')) * X(i, :)';
end

J = -1 * sumJ / m;
grad = -1 * sumG ./ m;






% =============================================================

end

%Give theta and xVals both as column vectors of equal dimension
function H = hyp(theta, xVals)
	H = sigmoid(sum(theta' * xVals));
end