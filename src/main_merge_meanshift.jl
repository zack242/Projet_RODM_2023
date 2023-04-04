include("building_tree.jl")
include("utilities.jl")
include("merge.jl")

using Distances
using LinearAlgebra


function main_merge()
    for dataSetName in ["iris", "seeds", "wine"]
        
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

"""
Regroupe des données en utilisant l'algorithme Mean Shift avec un noyau gaussien

Entrées :
- x : caractéristiques des données
- bandwidth : la largeur du noyau gaussien

Sorties :
- un tableau de Cluster constituant une partition de x
"""
function meanShift(X,y, bandwidth) 
    n = length(y)
    m = length(X[1, :])

    # Initialisation des clusters
    clusters = Vector{Cluster}([])

    # Lister les données non encore affectées à un cluster
    unassigned = Set(1:n)

    # Tant qu'il reste des données non affectées
    while length(unassigned) > 1
        # Choisir une donnée au hasard
        id = rand(unassigned)

        # Créer un nouveau cluster avec cette donnée
        center = X[id, :]
        cluster = Cluster(id, X, y)

        # Ajouter ce cluster à la liste des clusters
        push!(clusters, cluster)

        # Retirer cette donnée de la liste des données non affectées
        delete!(unassigned, id)
        #println("Cluster ", length(clusters), " : ", length(unassigned), " data remaining\r")
        # Tant qu'il y'a des données qui se rapporche du center du cluster
        while true 
            # Calcul de la distance de chaque point non affecté au centre du cluster
            distances = [euclidean(X[i, :], center) for i in unassigned]
            remaining = collect(unassigned)
            #add gaussian 
            weights = exp.(-distances .^ 2 ./ bandwidth)
            totalWeight = sum(weights)

            if totalWeight == 0
                break
            end
            # Calcul de la nouvelle moyenne pondérée des points proches
            newCenter = (weights' * X[remaining, :]) ./ totalWeight

            # Si le nouveau centre est suffisamment proche du centre précédent
            if LinearAlgebra.norm(newCenter' - center) < 1e-3
                break
            end
            center = newCenter'
        end 
        #cluster.class = mode(y[cluster.dataIds])
        assigned = collect(setdiff(1:n, unassigned)) 
        cluster.x = X[assigned, :]
       
    end
    #println(length(clusters), " clusters\t")
    return clusters
end

function testMerge(X_train, Y_train, X_test, Y_test, D, classes; time_limit::Int=-1, isMultivariate::Bool = false)

    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    for gamma in 0:1
        print("\t\t\t", gamma * 100, "%\t\t")
        clusters = meanShift(X_train, Y_train, 1.5)
        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap = build_tree(clusters, D, classes, multivariate = isMultivariate, time_limit = time_limit)
        print(round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        println(round(resolution_time, digits=1), "s")
    end
    println() 
end 
