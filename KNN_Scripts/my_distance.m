function [d] =  my_distance(x_1, x_2, params)
%MY_DISTANCE Computes the distance between two datapoints (as column vectors)
%   depending on the choosen distance type={'L1','L2','LInf'}
%
%   input -----------------------------------------------------------------
%   
%       o x_1   : (N x 1),  N-dimensional datapoint
%       o x_2   : (N x 1),  N-dimensional datapoint
%       o params : struct array containing the parameters of the KNN (k,
%                  d_type and eventually the parameters for the Gower
%                  similarity measure)
%
%   output ----------------------------------------------------------------
%
%       o d      : distance between x_1 and x_2 depending on distance
%                  type {'L1','L2','LInf'}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = 0;
switch params.d_type
    case 'L1'  % Manhattan Distance
        d = norm(x_1 - x_2, 1);
        % or
        % d = sum(abs(x_1-x_2));
        
    case 'L2' % Euclidean Distance        
%         d = norm(x_1 - x_2, 2);        
        % or        
        d = sqrt( sum((x_1 - x_2).^2) ); 
        % or
        % diff = x_1 - x_2;
        % d = sqrt(diff * diff');
        
    case 'LInf' % Infinity Norm      
%         d = norm(x_1 - x_2, Inf);        
        % or
        d = max(abs(x_1-x_2));
        
    case 'Gower'
        d = 1-gower_similarity(x_1,x_2,params.data_type,params.rk);
    
    otherwise
        warning('Unexpected distance type. No distance computed.')
end



end