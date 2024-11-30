#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#define db double
#define ll long long
#define sz(x) (ll) (sizeof(x) / sizeof(x[0]))
#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)


// Model configuration
ll epochs = 1000;
ll batch_size = 2;
db lr = 0.01;
ll nodes_per_layer[] = {0, 4, 4, 1};
char *activation_per_layer[] = {"#", "relu", "relu", "softmax"};
char *loss_function = "categorical_crossentropy";


// Model parameters (weight, bias, etc.)
ll m, n, num_of_classes = 0;
db in[5005][5005], out[5005], batch_in[5005][5005], batch_out[5005];
db W[5005][5005], B[5005][5005], Y[5005][5005], A[5005][5005];
db dW[5005][5005], dB[5005][5005], dA[5005][5005];
ll class[5005];


// Activation function
db sigmoid(db x) {return 1 / (1 + exp(-x));}
db sigmoid_dx(db x) {return sigmoid(x) * (1 - sigmoid(x));}
db relu(db x) {return x > 0 ? x : 0;}
db relu_dx(db x) {return x > 0;}
db tanh(db x) {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));}
db tanh_dx(db x) {return 1 - tanh(x) * tanh(x);}
db linear(db x) {return x;}
db linear_dx(db x) {return 1;}
db softmax(db x, ll layer, ll node) {
    db total_exp = 0;
    for(ll cnode=0;cnode<nodes_per_layer[layer];cnode++) {
        total_exp += exp(Y[layer][cnode]);
    }
    return exp(x) / total_exp;
}

db activation_fn(db x, ll layer, ll node) {
    if(activation_per_layer[layer] == "relu") return relu(x);
    else if(activation_per_layer[layer] == "sigmoid") return sigmoid(x);
    else if(activation_per_layer[layer] == "tanh") return tanh(x);
    else if(activation_per_layer[layer] == "softmax") return softmax(x, layer, node);
    else return linear(x);
}
db activation_fn_dx(db x, ll layer, ll node) {
    if(activation_per_layer[layer] == "relu") return relu_dx(x);
    else if(activation_per_layer[layer] == "sigmoid") return sigmoid_dx(x);
    else if(activation_per_layer[layer] == "tanh") return tanh_dx(x);
    else return linear_dx(x);
}


// One-hot encoding
int cmp(const void *a, const void *b) {
    ll *x = (ll *) a, *y = (ll *) b;
    return *x > *y;
}
void one_hot() {
    for(ll i=0;i<m;i++) {
        bool is_duplicate = 0;

        for(ll j=0;j<num_of_classes;j++) {
            if(out[i] == class[j]) {
                is_duplicate = 1;
                break;
            }
        }

        if(!is_duplicate) class[num_of_classes++] = out[i];
    }

    qsort(class, num_of_classes, sizeof(class[0]), cmp);
}
int find_one_hot_pos(db x) {
    for(ll i=0;i<num_of_classes;i++) {
        if(class[i] == x) return i;
    }
    return 0;
}


// Get position of max value
ll argmax(ll layer) {
    db ma = A[layer][0]; ll pos = 0;
    for(ll i=0;i<nodes_per_layer[layer];i++) {
        if(A[layer][i] > ma) {ma = A[layer][i]; pos = i;}
    }
    return pos;
}


// Get position of min value
ll argmin(ll layer) {
    db mi = A[layer][0]; ll pos = 0;
    for(ll i=0;i<nodes_per_layer[layer];i++) {
        if(A[layer][i] < mi) {mi = A[layer][i]; pos = i;}
    }
    return pos;
}


// Loss function and gradient
db loss_fn(db truth, db predict) {
    if(loss_function == "binary_crossentropy") {
        return -(truth) * log(predict) - (1 - truth) * log(1 - predict);
    }
    else if(loss_function == "categorical_crossentropy") {
        return -log(A[sz(nodes_per_layer) - 1][find_one_hot_pos(truth)]);
    }
    else if(loss_function == "mse") {
        return pow(truth - predict, 2);
    }
    else return 0;
}
void loss_fn_dx(db truth, db predict) {
    if(loss_function == "binary_crossentropy") {
        dA[sz(nodes_per_layer)-1][0] = -truth / predict + (1 - truth) / (1 - predict);
    }
    else if(loss_function == "categorical_crossentropy") {
        for(ll node=0;node<num_of_classes;node++) {
            dA[sz(nodes_per_layer)-1][node] = A[sz(nodes_per_layer)-1][node] - (node == find_one_hot_pos(truth));
        }
    }
    else if(loss_function == "mse") {
        dA[sz(nodes_per_layer)-1][0] = 2 * (predict - truth);
    }
    else return;
}


// Generate random number between -1 and 1
db r2() {return ((db) rand() / (db) RAND_MAX) * 2 - 1;}


// Reset parameter
void reset_forward() {
    memset(A, 0, sizeof(A));
    memset(Y, 0, sizeof(Y));
}

void reset_backward() {
    memset(dW, 0, sizeof(dW));
    memset(dB, 0, sizeof(dB));
    memset(dA, 0, sizeof(dA));
}


// Create batch
void swap(db *a, db *b) {
    db temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle_array() {
    for(ll i=m-1;i>=0;i--) {
        ll new_pos = rand() % (i + 1);

        for(ll j=0;j<n;j++) swap(&in[i][j], &in[new_pos][j]);
        swap(&out[i], &out[new_pos]);
    }
}

ll current_pos = 0;
ll create_batch() {
    if(current_pos >= m) {
        current_pos = 0;
        shuffle_array();
        return 0;
    }

    ll local_pos = 0;
    for(ll i=current_pos;i<min(current_pos + batch_size, m);i++) {
        for(ll j=0;j<n;j++) batch_in[local_pos][j] = in[i][j];
        batch_out[local_pos] = out[i];
        local_pos++;
    }
    current_pos = min(current_pos + batch_size, m);

    return local_pos;
}


// Forward propagation
void forward() {
    for(ll layer=1;layer<sz(nodes_per_layer);layer++) {
        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
                Y[layer][node] += W[layer][w_order] * A[layer-1][pre_node];

                w_order++;
            }

            Y[layer][node] += B[layer][node];
        }

        for(ll node=0;node<nodes_per_layer[layer];node++) {
            A[layer][node] = activation_fn(Y[layer][node], layer, node);
        }

        //Print result
        // printf("######## Layer %lld ########\n", layer);
        // w_order = 0;
        // for(ll node=0;node<nodes_per_layer[layer];node++) {\
        //     printf("--- Node %lld ---\n", node);
        //     printf("w: ");
        //     for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
        //         printf("%lf ", W[layer][w_order++]);
        //     }

        //     printf("\nb: %lf\ny: %lf\na: %lf\n", B[layer][node], Y[layer][node], A[layer][node]);
        // }
        // printf("\n");
    }
}


// Backward propagation
void backward(db truth, db predict) {
    memset(dA, 0, sizeof(dA));

    loss_fn_dx(truth, predict);

    for(ll layer=sz(nodes_per_layer)-1;layer>=1;layer--) {
        // printf("######## Layer %lld ########\n", layer);

        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            // printf("--- Node %lld ---\n", node);
            // printf("dW: ");

            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
                dW[layer][w_order] += dA[layer][node] * (activation_per_layer[layer] != "softmax" ? activation_fn_dx(Y[layer][node], layer, node) : 1) * A[layer-1][pre_node];

                dA[layer-1][pre_node] += dA[layer][node] * (activation_per_layer[layer] != "softmax" ? activation_fn_dx(Y[layer][node], layer, node) : 1) * W[layer][w_order];

                // printf("%lf ", dW[layer][w_order]);

                w_order++;
            }

            dB[layer][node] += dA[layer][node] * (activation_per_layer[layer] != "softmax" ? activation_fn_dx(Y[layer][node], layer, node) : 1);

            // printf("\ndB: %lf\ndA: %lf\n", dB[layer][node], dA[layer][node]);
        }
        // printf("\n");
    }
}


// Apply gradient descent
void gradient_descent(ll curr_batch_size) {
    for(ll layer=1;layer<sz(nodes_per_layer);layer++) {
        // printf("######## Layer %lld ########\n", layer);
        // printf("New W: ");
        for(ll w_order=0;w_order<nodes_per_layer[layer]*nodes_per_layer[layer-1];w_order++) {
            W[layer][w_order] -= lr * (dW[layer][w_order] / curr_batch_size);

            // printf("%lf ", W[layer][w_order]);
        }

        // printf("\nNew B: ");
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            B[layer][node] -= lr * (dB[layer][node] / curr_batch_size);

            // printf("%lf ", B[layer][node]);
        }
        // printf("\n");
    }
}


int main() {
    freopen("MAIN.INP", "r", stdin); //freopen("MAIN.OUT", "w", stdout);

	scanf("%lld %lld", &m, &n);
	for(ll i=0;i<m;i++) {
        for(ll j=0;j<n;j++) scanf("%lf", &in[i][j]);
        scanf("%lf", &out[i]);
    }

    if(loss_function == "categorical_crossentropy") {
        one_hot();
        nodes_per_layer[sz(nodes_per_layer) - 1] = num_of_classes;
    }
    
    // Set input size
    nodes_per_layer[0] = n;

    // Random weight and bias
    srand(852006);
    
    for(ll layer=1;layer<sz(nodes_per_layer);layer++) {
        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) W[layer][w_order++] = r2();
            B[layer][node] = r2();
        }
    }

    // Training
    for(ll epoch=1;epoch<=epochs;epoch++) {
        printf("+++++++++ EPOCH %lld +++++++++\n", epoch);

        ll curr_batch_size, batch_num = 0;
        db total_loss = 0;

        // Load through dataset
        while((curr_batch_size = create_batch()) > 0) {
            // printf("Size: %lld\n", curr_batch_size);
            
            // Reset parameter
            db loss = 0;
            reset_forward(); reset_backward();

            for(ll i=0;i<curr_batch_size;i++) {
                // Reset parameters
                reset_forward();

                // Set input and output
                for(ll j=0;j<n;j++) A[0][j] = batch_in[i][j];
                db truth = batch_out[i];

                // Forward propagation
                // printf("********** Forward **********\n");
                forward();
                
                // Calculate loss
                loss += loss_fn(truth, A[sz(nodes_per_layer)-1][0]);

                // Backward propagation
                // printf("********** Backward **********\n");
                backward(truth, A[sz(nodes_per_layer)-1][0]);
            }

            // Calculate total loss per epoch and print
            total_loss += loss / curr_batch_size;
            batch_num++;
            printf("Batch %lld - Loss: %.9lf\n", batch_num, total_loss / batch_num);
            fflush(stdout);


            // Apply gradient descent
            gradient_descent(curr_batch_size);

            // delay(1);
        }
        // printf("++++++++++++ END OF EPOCH %lld ++++++++++++\n\n\n", epoch);
    }

    // Test model
    printf("************* TEST MODEL *************\n");

    ll testcase; scanf("%lld", &testcase);
    for(ll i=1;i<=testcase;i++) {
        // Reset parameter
        reset_forward();

        for(ll j=0;j<n;j++) scanf("%lf", &A[0][j]);

        // Predict
        forward();

        printf("Testcase #%lld - Predict value: ", i);
        if(loss_function == "categorical_crossentropy") printf("%lld", class[argmax(sz(nodes_per_layer)-1)]);
        else if(loss_function == "binary_crossentropy") printf("%lld", A[sz(nodes_per_layer)-1][0] > 0.5);
        else printf("%lf", A[sz(nodes_per_layer)-1][0]);
        printf("\n");
    }
}
