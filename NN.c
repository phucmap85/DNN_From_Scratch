#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

#define db double
#define ll long long
#define sz(x) (ll) (sizeof(x) / sizeof(x[0]))


// Model configuration
ll epochs = 60;
ll batch_size = 2;
db lr = 0.1;
ll nodes_per_layer[] = {0, 4, 4, 1};
char *activation_per_layer[] = {"#", "relu", "relu", "sigmoid"};


// Model parameters (weight, bias, etc.)
db in[1005][1005], out[1005], truth = 0;
db W[1005][1005], B[1005][1005], Y[1005][1005], A[1005][1005];
db dW[1005][1005], dB[1005][1005], dA[1005][1005];


// Activation function
db sigmoid(db x) {return 1 / (1 + exp(-x));}
db sigmoid_dx(db x) {return sigmoid(x) * (1 - sigmoid(x));}
db relu(db x) {return x > 0 ? x : 0;}
db relu_dx(db x) {return x > 0;}
db tanh(db x) {return (exp(x) - exp(-x)) / (exp(x) + exp(-x));}
db tanh_dx(db x) {return 1 - tanh(x) * tanh(x);}
db linear(db x) {return x;}
db linear_dx(db x) {return 1;}

db activation_fn(db x, ll layer) {
    if(activation_per_layer[layer] == "relu") return relu(x);
    else if(activation_per_layer[layer] == "sigmoid") return sigmoid(x);
    else if(activation_per_layer[layer] == "tanh") return tanh(x);
    else return linear(x);
}
db activation_fn_dx(db x, ll layer) {
    if(activation_per_layer[layer] == "relu") return relu_dx(x);
    else if(activation_per_layer[layer] == "sigmoid") return sigmoid_dx(x);
    else if(activation_per_layer[layer] == "tanh") return tanh_dx(x);
    else return linear_dx(x);
}


// Delay
void delay(ll number_of_seconds) {
    // Converting time into milli_seconds
    ll milli_seconds = 1000 * number_of_seconds;

    // Storing start time
    clock_t start_time = clock();

    // looping till required time is not achieved
    while (clock() < start_time + milli_seconds);
}


// Reset parameter
void reset() {
    memset(A, 0, sizeof(A)); memset(Y, 0, sizeof(Y));
    memset(dW, 0, sizeof(dW)); memset(dB, 0, sizeof(dB)); memset(dA, 0, sizeof(dA));
}


// Generate random number between -1 and 1
db r2() {return ((db) rand() / (db) RAND_MAX) * 2 - 1;}


// Forward propagation
void forward() {
    for(ll layer=1;layer<sz(nodes_per_layer);layer++) {
        // printf("######## Layer %lld ########\n", layer);
        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            // printf("--- Node %lld ---\n", node);
            // printf("w: ");

            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
                Y[layer][node] += W[layer][w_order] * A[layer-1][pre_node];

                // printf("%lf ", W[layer][w_order]);

                w_order++;
            }

            Y[layer][node] += B[layer][node];
            A[layer][node] = activation_fn(Y[layer][node], layer);

            // printf("\nb: %lf\ny: %lf\na: %lf\n", B[layer][node], Y[layer][node], A[layer][node]);
        }
        // printf("\n");
    }
}


// Backward propagation
void backward() {
    dA[sz(nodes_per_layer)-1][0] = -truth / A[sz(nodes_per_layer)-1][0] + (1 - truth) / (1 - A[sz(nodes_per_layer)-1][0]);
    // printf("%lf\n", dA[sz(nodes_per_layer)-1][0]);

    for(ll layer=sz(nodes_per_layer)-1;layer>=1;layer--) {
        // printf("######## Layer %lld ########\n", layer);

        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            // printf("--- Node %lld ---\n", node);
            // printf("dW: ");

            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
                dW[layer][w_order] = dA[layer][node] * activation_fn_dx(Y[layer][node], layer) * A[layer-1][pre_node];

                dA[layer-1][pre_node] += dA[layer][node] * activation_fn_dx(Y[layer][node], layer) * W[layer][w_order];

                // printf("%lf ", dW[layer][w_order]);

                w_order++;
            }

            dB[layer][node] = dA[layer][node] * activation_fn_dx(Y[layer][node], layer);

            // printf("\ndB: %lf\ndA: %lf\n", dB[layer][node], dA[layer][node]);
        }
        // printf("\n");
    }
}


// Apply gradient descent
void gradient_descent() {
    for(ll layer=1;layer<sz(nodes_per_layer);layer++) {
        // printf("######## Layer %lld ########\n", layer);
        // printf("New W: ");
        for(ll w_order=0;w_order<nodes_per_layer[layer]*nodes_per_layer[layer-1];w_order++) {
            W[layer][w_order] -= lr * dW[layer][w_order];

            // printf("%lf ", W[layer][w_order]);
        }

        // printf("\nNew B: ");
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            B[layer][node] -= lr * dB[layer][node];

            // printf("%lf ", B[layer][node]);
        }
        // printf("\n");
    }
}


int main() {
    freopen("MAIN.INP", "r", stdin); freopen("MAIN.OUT", "w", stdout);

	ll m, n; scanf("%lld %lld", &m, &n);
	for(ll i=0;i<m;i++) {
        for(ll j=0;j<n;j++) scanf("%lf", &in[i][j]);
        scanf("%lf", &out[i]);
    }
    
    nodes_per_layer[0] = n;

    // Random weight and bias
    srand(21042006);
    
    for(ll layer=1;layer<sz(nodes_per_layer);layer++) {
        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) W[layer][w_order++] = r2();
            B[layer][node] = r2();
        }
    }

    // Training
    for(ll epoch=0;epoch<epochs;epoch++) {
        printf("++++++++++++ EPOCH %lld ++++++++++++\n\n", epoch + 1);

        // Load through dataset
        for(ll i=0;i<m;i++) {
            // Reset parameters
            reset();

            // Set input and output
            for(ll j=0;j<n;j++) A[0][j] = in[i][j];
            truth = out[i];

            // Forward propagation
            // printf("********** Forward **********\n");
            forward();

            // Calculate loss and print
            db loss = -truth * log(A[sz(nodes_per_layer)-1][0]) - (1 - truth) * log(1 - A[sz(nodes_per_layer)-1][0]);
            printf("====> Loss: %lf\n\n", loss);
            fflush(stdout);
            
            // Backward propagation
            // printf("********** Backward **********\n");
            backward();

            // Apply gradient descent
            gradient_descent();

            // delay(1);
        }
        printf("++++++++++++ END OF EPOCH %lld ++++++++++++\n\n\n", epoch + 1);
    }

    // Test model
    printf("************* TEST MODEL *************\n");

    ll testcase; scanf("%lld", &testcase);
    for(ll i=1;i<=testcase;i++) {
        // Reset parameter
        reset();

        for(ll j=0;j<n;j++) scanf("%lf", &A[0][j]);

        // Predict
        forward();

        printf("Testcase #%lld - Predict value: %.9lf\n", i, A[sz(nodes_per_layer)-1][0]);
    }
}
