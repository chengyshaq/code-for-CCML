function [ccml_model, loss] = CCML( X, Y, optmParameter)
%% optimization parameters
alpha            = optmParameter.alpha;
beta             = optmParameter.beta;
theta            = optmParameter.theta;
maxIter          = optmParameter.maxIter;
miniLossMargin   = optmParameter.minimumLossMargin;
%% initializtion
num_dim= size(X,2);
XTX = X'*X;

% do causal inference
loss = NaN();
V_1 = causalInference(Y);
V_2 = V_1';

P = 1 - pdist2(Y'+eps,Y'+eps,'cosine');
[row, col] = find(V_2 == 1);
loopSize = size(row);
for i = 1 : loopSize
    P(row(i), col(i)) = 0;
end
% L = diag(sum(P, 2)) - P;
L = P;

iter    = 1;
bk = 1;
bk_1 = 1;
oldloss = 0;
%% proximal gradient
while iter <= maxIter
    %%fix A and Q solve W
    norm_1=norm(XTX)^2;
    W_s   = (XTX + theta*eye(num_dim)) \ (X'*Y);
    W_s_1 = W_s;
    Lip = sqrt(2*(norm_1) + norm(alpha*L)^2);
    
    
    W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
    
    Gw_s_k = W_s_k - 1/Lip * ((XTX*W_s_k - X'*Y) + alpha*W_s_k*L);
    bk_1   = bk;
    bk     = (1 + sqrt(4*bk^2 + 1))/2;
    W_s    = softthres(Gw_s_k,beta/Lip);
    predictionLoss = 0.5*trace((X*W_s - Y)'*(X*W_s_1 - Y))+0.5*alpha*trace(W_s*L*W_s');
    spares_Ws_Loss = sum(sum(W_s~=0));
    totalloss = predictionLoss +  beta*spares_Ws_Loss;
    loss = [loss, totalloss];
    
    if abs(oldloss - totalloss) <= miniLossMargin
        
        break;
    elseif totalloss <=0
        break;
    else
        oldloss = totalloss;
    end
    if iter>maxIter
        
    end
    iter=iter+1;
end

ccml_model.W=W_s;
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
W = max(W_t-lambda,0) - max(-W_t-lambda,0);
end
