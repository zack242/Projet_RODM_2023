include("utilities.jl")
include("merge.jl")

using Distances
using LinearAlgebra
using Statistics

"""
Algorithme de clustering hiérarchique

Entrées :
- x : caractéristiques des données d'entraînement
- y : étiquettes des données d'entraînement
- type : type de liaison utilisé
"""

function agglomerative_cluster(x, y, type,nbr_clusters=1)
    # Calculate number of data points and features
    n = length(y)
    m = length(x[1, :])

    # Initialize a vector of clusters with each data point as a separate cluster
    clusters = Vector{Cluster}([])
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Merge the two closest clusters iteratively until only one cluster remains
    while length(clusters) > nbr_clusters
        # Find the two closest clusters
        minDist = Inf
        c1 = 0
        c2 = 0
        for i in 1:length(clusters)
            for j in i+1:length(clusters)
                if type == "Single"
                    dist = singleLinkage(clusters[i], clusters[j], x)
                elseif type == "Average"
                    dist = averageLinkage(clusters[i], clusters[j], x)
                elseif type == "Complet"
                    dist = completeLinkage(clusters[i], clusters[j], x)
                end

                if dist < minDist
                    minDist = dist
                    c1 = i
                    c2 = j
                end
            end
        end
        # Merge the two closest clusters
        merge!(clusters[c1], clusters[c2])
        # Remove the second cluster
        deleteat!(clusters, c2)
    end

    # Return the final set of clusters
    return clusters
end


"""
Algorithme de clustering mean shift

Entrées :
- x : caractéristiques des données
- y : cluster des données
- bandwidth : la largeur du noyau gaussien

"""
function meanShift(X, y, bandwidth)
    n = length(y)
    m = length(X[1, :])

    # Initialize clusters
    clusters = Vector{Cluster}([])

    # List unassigned data
    unassigned = Set(1:n)

    # While there is unassigned data
    while length(unassigned) > 1
        # Choose a random data point
        id = rand(unassigned)

        # Create a new cluster with this data point
        center = X[id, :]
        cluster = Cluster(id, X, y)

        # Add this cluster to the list of clusters
        push!(clusters, cluster)

        # Remove this data point from the list of unassigned data
        delete!(unassigned, id)

        # While there are data points that are close to the cluster center
        while true
            # Compute the distance of each unassigned point to the cluster center
            distances = [euclidean(X[i, :], center) for i in unassigned]
            remaining = collect(unassigned)

            # Add Gaussian weights
            weights = exp.(-distances .^ 2 ./ bandwidth)
            totalWeight = sum(weights)

            if totalWeight == 0
                break
            end

            # Compute the new weighted mean of nearby points
            newCenter = (weights' * X[remaining, :]) ./ totalWeight

            # If the new center is close enough to the previous center
            if LinearAlgebra.norm(newCenter' - center) < 1e-3
                break
            end
            center = newCenter'
        end 
        # Assign data points to nearest centroid
        assigned = Int[]
        remaining = collect(unassigned)
        for i in remaining
            if euclidean(X[i, :], center) < 0.5
                # Add data point to cluster
                push!(cluster.dataIds, i)
                push!(assigned, i)
            end
        end  
        
        # Mark assigned data points as assigned
        setdiff!(unassigned, assigned)

    end
    return clusters
end



"""
Algorithme de clustering DBSCAN

Entrées :
- x : caractéristiques des données
- eps : distance maximale entre deux points pour être considérés comme voisins
- min_samples : nombre minimal de voisins pour qu'un point soit considéré comme un centre de cluster

"""


function my_dbscan(X::Matrix{T}, eps::T; min_samples::Int=5) where {T<:AbstractFloat}
    n = size(X, 1)
    labels = zeros(Int, n)
    visited = zeros(Bool, n)

    # Compute pairwise distances between points
    D = pairwise(Euclidean(), X, dims=1)

    C = 0
    for i in 1:n
        if visited[i]
            continue
        end
        visited[i] = true

        # Get the indices of all points within eps distance of point i
        neighbours = findall(D[:, i] .<= eps)

        if length(neighbours) < min_samples
            # Point i is noise
            labels[i] = 0
        else
            # Expand the cluster
            C += 1
            expand_cluster!(i, neighbours, C, labels, visited, D, eps, min_samples)
        end
    end

    return labels
end


### Utils functions used in the clustering algorithms ###

"""
Expand the cluster from point i

Entrées :
- i : point de départ
- neighbours : indices des voisins de i
- C : numéro du cluster
- labels : vecteur des étiquettes
- visited : vecteur des points visités
- D : matrice des distances

"""

function expand_cluster!(i, neighbours, C, labels, visited, D, eps, min_samples)
    labels[i] = C

    while !isempty(neighbours)
        j = neighbours[1]
        deleteat!(neighbours, 1)

        if !visited[j]
            visited[j] = true
            neighbours_j = findall(D[:, j] .<= eps)
            if length(neighbours_j) >= min_samples
                neighbours = union(neighbours, neighbours_j)
            end
        end

        if labels[j] == 0
            labels[j] = C
        end
    end
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
            dist = euclidean(x[i, :], x[j, :])
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
            dist = euclidean(x[i, :], x[j, :])
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
            sumDist += euclidean(x[i, :], x[j, :])
        end
    end
    return sumDist / (length(c1.dataIds) * length(c2.dataIds))
end



