function hat_y = predict( X, node )
    if ~isempty( node.h )
        hat_y = node.h;
        return 
    end

    hat_y = zeros( size(X, 1), 1 );

    left_idx = X(:, node.j) < node.theta;
    right_idx = ~left_idx;
    hat_y(left_idx) = predict( X(left_idx, :), node.left );
    hat_y(right_idx) = predict( X(right_idx, :), node.right );