include("building_tree.jl")
include("utilities.jl")
include("merge.jl")

function main_merge_agglomeration()
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

        for D in 2:3
            println("\tD = ", D)
            println("\t\tUnivarié")
            testMerge(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = false)
            println("\t\tMultivarié")
            testMerge(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true)
        end
    end
end 

function agglomerative_cluster(x,y,type)
    n = length(y)
    m = length(x[1,:])

    #Initialize a empty vector of clusters
    clusters = Vector{Cluster}([])
    #Initialize all the point as clusters 
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId,x,y))
    end

    #print("Initialisation : ", length(clusters), " clusters\t")

    #While there is more than one cluster
    while length(clusters) > 1
        #Find the two closest clusters
        minDist = Inf
        c1 = 0
        c2 = 0
        for i in 1:length(clusters)
            for j in i+1:length(clusters)
                if type == "Single"
                    dist = singleLinkage(clusters[i],clusters[j],x)
                elseif type == "Average"
                    dist = averageLinkage(clusters[i],clusters[j],x)
                elseif type == "Complet"
                    dist = completeLinkage(clusters[i],clusters[j],x)
                end

                if dist < minDist
                    minDist = dist
                    c1 = i
                    c2 = j
                end
            end
        end
        #Merge the two closest clusters
        merge!(clusters[c1],clusters[c2])
        #Remove the second cluster
        deleteat!(clusters,c2)
        
    end

    return clusters

end 

"""
Calcule la distance entre deux clusters pour la méthode de liaison simple

Entrées :
- c1 : cluster 1
- c2 : cluster 2
- x  : caractéristique des données d'entraînement
"""
function singleLinkage(c1::Cluster, c2::Cluster, x::Matrix{Float64})
    minDist = Inf
    for i in c1.dataIds
        for j in c2.dataIds
            dist = euclidean(x[i,:], x[j,:])
            if dist < minDist
                minDist = dist
            end
        end
    end
    return minDist
end

"""
Calcule la distance entre deux clusters pour la méthode de liaison complète

Entrées :
- c1 : cluster 1
- c2 : cluster 2
- x  : caractéristique des données d'entraînement
"""

function completeLinkage(c1::Cluster, c2::Cluster, x::Matrix{Float64})
    maxDist = -Inf
    for i in c1.dataIds
        for j in c2.dataIds
            dist = euclidean(x[i,:], x[j,:])
            if dist > maxDist
                maxDist = dist
            end
        end
    end
    return maxDist
end

"""
Calculer la distance entre deux clusters pour la méthode de liaison moyenne

Entrées :
- c1 : cluster 1
- c2 : cluster 2
- x  : caractéristique des données d'entraînement
"""

function averageLinkage(c1::Cluster, c2::Cluster, x::Matrix{Float64})
    sumDist = 0
    for i in c1.dataIds
        for j in c2.dataIds
            sumDist += euclidean(x[i,:], x[j,:])
        end
    end
    return sumDist/(length(c1.dataIds)*length(c2.dataIds))
end

function testMerge(X_train, Y_train, X_test, Y_test, D, classes; time_limit::Int=-1, isMultivariate::Bool = false)

    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    for type in ["Single","Average","Complet"]
        print("\t\t\t",type, "\t\t")
        clusters = agglomerative_cluster(X_train, Y_train,type)
        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap = build_tree(clusters, D, classes, multivariate = isMultivariate, time_limit = time_limit)
        print(round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        println(round(resolution_time, digits=1), "s")
    end
    println() 
end 
