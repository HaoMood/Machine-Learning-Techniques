function [ node, totInNodes ] = decisionTree(X, y, totInNodes)
    node.j = [];
    node.theta = [];
    node.left = [];
    node.right = [];
    node.h = [];

    [ m, n ] = size( X );

    if size( unique( y ), 1 ) == 1 
        node.h = unique( y );
        return
    end

    if isequal( ones( m, 1 ) * X(1, :), X )
        y_uniq = unique( y );
        [ ~, max_count ]  = max( hist( y, y_uniq ) );
        node.h = y_uniq( max_count );
        return
    end

    totInNodes = totInNodes + 1;

    [ node.j, node.theta ] = decisionStump( X, y );

    left_idx = X(:, node.j) < node.theta;
    right_idx = ~left_idx;
    [ node.left totInNodes ] = decisionTree( X(left_idx, :), y(left_idx, :), totInNodes );
    [ node.right totInNodes ] = decisionTree( X(right_idx, :), y(right_idx, :), totInNodes );

function [ opt_j, opt_theta ] = decisionStump( X, y );
    opt_j = [];
    opt_theta = [];
    opt_J = 1e4;

    [ m, n ] = size( X );

    for j = 1: n
        xj = unique( sort( X(:, j) )' );
        theta = [ ( xj(1: end - 1) + xj(2: end) ) / 2 ];
        size_theta = size( theta, 2 );
        left_idx = ( X(:, j) * ones( 1, size_theta ) - ones( m, 1 ) * theta ) < 0;
        right_idx = ~left_idx;
        y_left = y * ones( 1, size_theta ) .* left_idx;
        y_right = y * ones( 1, size_theta ) .* right_idx;
        J = sum( left_idx ) .* gini( y_left ) + sum( right_idx ) .* gini( y_right );
        [ min_J, idx_min ] = min( J );
        if ( min_J < opt_J )
            opt_J = min_J;
            opt_theta = theta(idx_min);
            opt_j = j;
        end
    end

function idx = gini( y )
    [ m, size_theta ] = size( y );
    idx = zeros( 1, size_theta );
    for k = 1: size_theta
        yk = y(:, k);
        yk(yk == 0) = [];
        mp = size( yk, 1 );
        if size( unique( yk ), 1 ) == 1
            idx(k) = 1 - ( size( yk, 1 ) / mp )^2;
        else
            y_count = hist( yk, unique( yk ) );
            idx(k) = 1 - sum( ( y_count / mp ).^2 );
        end
    end