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
ll num_of_layer, *nodes_per_layer;
char *activation_per_layer[105];
char loss_function[105];


// Model parameters (weight, bias, etc.)
ll m, n, num_of_classes = 1;
db W[1005][20005], B[1005][20005], Y[1005][20005], A[1005][20005];
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
    if(strcmp(activation_per_layer[layer], "relu") == 0) return relu(x);
    else if(strcmp(activation_per_layer[layer], "sigmoid") == 0) return sigmoid(x);
    else if(strcmp(activation_per_layer[layer], "tanh") == 0) return tanh(x);
    else if(strcmp(activation_per_layer[layer], "softmax") == 0) return softmax(x, layer, node);
    else return linear(x);
}
db activation_fn_dx(db x, ll layer, ll node) {
    if(strcmp(activation_per_layer[layer], "relu") == 0) return relu_dx(x);
    else if(strcmp(activation_per_layer[layer], "sigmoid") == 0) return sigmoid_dx(x);
    else if(strcmp(activation_per_layer[layer], "tanh") == 0) return tanh_dx(x);
    else return linear_dx(x);
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


// Forward propagation
void forward() {
    for(ll layer=1;layer<num_of_layer;layer++) {
        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
                Y[layer][node] += W[layer][w_order++] * A[layer-1][pre_node];
            }

            Y[layer][node] += B[layer][node];
        }

        for(ll node=0;node<nodes_per_layer[layer];node++) {
            A[layer][node] = activation_fn(Y[layer][node], layer, node);
        }

        // Print result
        printf("######## Layer %lld ########\n", layer);
        w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {\
            printf("--- Node %lld ---\n", node);
            printf("w: ");
            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
                printf("%lf ", W[layer][w_order++]);
            }

            printf("\nb: %lf\ny: %lf\na: %lf\n", B[layer][node], Y[layer][node], A[layer][node]);
        }
        printf("\n");
    }
}


int main() {
    freopen("best.txt", "r", stdin); freopen("MAIN.OUT", "w", stdout);
    ll trash; scanf("%lld %lf %lld", &trash, &trash, &num_of_layer);

    nodes_per_layer = (ll *) malloc(num_of_layer * sizeof(ll));
    
    for(ll i=0;i<num_of_layer;i++) scanf("%lld", &nodes_per_layer[i]);
    for(ll i=0;i<num_of_layer;i++) {
        char t[305]; scanf("%s", &t);

        activation_per_layer[i] = (char *) malloc((strlen(t) + 5) * sizeof(char));

        strcpy(activation_per_layer[i], t);
    }

    scanf("%s", &loss_function);

    for(ll layer=1;layer<num_of_layer;layer++) {
        ll w_order = 0;
        for(ll node=0;node<nodes_per_layer[layer];node++) {
            for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
                scanf("%lf", &W[layer][w_order++]);
            }
        }

        for(ll node=0;node<nodes_per_layer[layer];node++) scanf("%lf", &B[layer][node]);

        // // Print result
        // printf("######## Layer %lld ########\n", layer);
        // w_order = 0;
        // for(ll node=0;node<nodes_per_layer[layer];node++) {\
        //     printf("--- Node %lld ---\n", node);
        //     printf("w: ");
        //     for(ll pre_node=0;pre_node<nodes_per_layer[layer-1];pre_node++) {
        //         printf("%lf ", W[layer][w_order++]);
        //     }

        //     printf("\nb: %lf\n", B[layer][node]);
        // }
        // printf("\n");
    }
    
    // Load dataset
    FILE *fptr;

    if((fptr = fopen("test.txt", "r")) == NULL) {
        printf("Error when reading file");
        exit(1);
    }

    for(ll i=0;i<nodes_per_layer[0];i++) fscanf(fptr, "%lf", &A[0][i]);

    // for(ll i=0;i<nodes_per_layer[0];i++) printf("%lf ", A[0][i]);

    fclose(fptr);
    
    // Forward propagation
    forward();

    db predict = A[num_of_layer-1][0];
    if(strcmp(loss_function, "categorical_crossentropy") == 0) predict = argmax(num_of_layer-1);
    else if(strcmp(loss_function, "binary_crossentropy") == 0) predict = A[num_of_layer-1][0] > 0.5;

    printf("Predicted value: %lf", predict);
}
