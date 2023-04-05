include("building_tree.jl")
include("utilities.jl")
include("merge.jl")

using Distances
using Statistics

function main_merge_DBSCAN()
    for dataSetName in ["iris", "seeds", "wine","diabetes", "ionosphere"]
        
        print("=== Dataset ", dataSetName)
        
        # Préparation des données
        include("./data/" * dataSetName * ".txt")
        
        # Ramener chaque caractéristique sur [0, 1]
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train,:]
        Y_train = Y[train]
        X_test = reducedX[test,:]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes        
        time_limit = 10

        for D in 2:4
            println("\tD = ", D)
            println("\t\tUnivarié")
            testMerge(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = false)
            println("\t\tMultivarié")
            testMerge(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true)
        end
    end
end 

function testMerge(X_train, Y_train, X_test, Y_test, D, classes; time_limit::Int=-1, isMultivariate::Bool = false)
    # DBSCAN clustering

    println("\t\t\tEpsilon\t\t# clusters\tGap")
    for epsilon in 0:0.3:1

        clustering = my_dbscan(X_train,epsilon, min_samples = 5)
        clusters = [findall(x -> x == i, clustering) for i in unique(clustering)]
        cluster_objects = [Cluster(i, X_train[cluster, :], Y_train[cluster]) for (i, cluster) in enumerate(clusters)]

        print("\t\t\t", epsilon, "\t\t")
        T, obj, resolution_time, gap = build_tree(cluster_objects, D, classes, multivariate=isMultivariate, time_limit=time_limit)
        print(length(clusters), " clusters\t")
        print(round(gap, digits=1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        println(round(resolution_time, digits=1), "s")
    end
    println() 
end

