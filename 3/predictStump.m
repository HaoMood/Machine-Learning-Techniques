function hat_y = predictStump( X, node )
    hat_y = zeros( size(X, 1), 1 );

    left_idx = X(:, node.j) < node.theta;
    right_idx = ~left_idx;
    hat_y(left_idx) = node.left.h;
    hat_y(right_idx) = node.right.h;