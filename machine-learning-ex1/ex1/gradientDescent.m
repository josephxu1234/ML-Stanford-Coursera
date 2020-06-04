function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	sum1 = 0;
	sum2 = 0;
	for i=1:m,
		sum1 += (hyp(theta, X(i, :)) - y(i, :));
		sum2 += (hyp(theta, X(i, :)) - y(i, :)) * X(i, 2);
	end
	
	theta = [theta(1, 1) - alpha / m * sum1; theta(2, 1) - alpha / m * sum2];



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

%theta is a n+1 by 1 vector
%x vals is a 1 by n+1 vector. should be 1 row of the X design matrix
function H = hyp(theta, xVals)
	H = theta' * xVals';
end