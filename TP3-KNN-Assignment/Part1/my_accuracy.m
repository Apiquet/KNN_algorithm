function [acc] =  my_accuracy(y_test, y_est)
%My_accuracy Computes the accuracy of a given classification estimate.
%   input -----------------------------------------------------------------
%   
%       o y_test  : (1 x M_test),  true labels from testing set
%       o y_est   : (1 x M_test),  estimated labes from testing set
%
%   output ----------------------------------------------------------------
%
%       o acc     : classifier accuracy
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Implement Eq. X here %%%%
N = length(y_test);
count = 0;
for i = 1:N
    if(y_test(i)==y_est(i)) count=count+1;
    end
end
acc = count/N;

end