#!/bin/sh

# FIXME: Balance the number of vectors per class
# FIXME: Make sure that the number of vectors per class modulo N is low (ideally 0)

# First argument to script is run tag
TAG=$1

# Second argument to script is number of folds for n-fold cross-validation
N=$2

# Clear the text files to for the cross-validation folds
for((i=1;i<N+1;++i)); do
    > ../../out/"$TAG"_$i.txt
done

# Build the feature vectors from the input sentences
./build_vectors.py ../../data/sentences.txt

# Get the paths to the files containing the vectors
FILES=$(find /home2/amkahn/workspace/RA/review-classifier/data/vec -maxdepth 1 -type f)

# Get the number of files/products
NUM=$(echo "$FILES" | wc -l)
echo "Number of classes:" $NUM

# Iterate through the class files, splitting data for each class randomly into N folds
for FILE in $FILES; do
    # Store the vectors of this class in an array
    readarray a < $FILE
    
    # Count the vectors in this class and determine the number of vectors per fold
    NUM=${#a[*]}
    echo $NUM "vectors in" $FILE
    K=$(($NUM / N))
    echo "Each fold for the first class will have this many vectors:" $K

    # Count the vectors in this class and determine the number of vectors per fold
#     NUM=$(cat "$FILE" | wc -l)
#     echo $NUM "vectors in" $FILE
#     K=$(($NUM / N))
#     echo "Each fold for the first class will have this many vectors:" $K

    # Get 1/Nth of the vectors from each class at random
    # Do this N-1 times
    for((i=1;i<N && ${#a[@]};++i)); do
        echo "Writing vectors for fold" $i

        # Create an array to hold N randomly selected vectors
        randf=()
    
        # Do this K times
        for((k=0;k<K && ${#a[@]};++k)); do
            # Pick random vector
            ((j=RANDOM%${#a[@]}))
            # Add to the array of random vectors
            randf+=( "${a[j]}" )
            # Remove from the array of all vectors
            a=( "${a[@]:0:j}" "${a[@]:j+1}" )
        done

        # Append random vectors to txt file for this fold
        for f in "${randf[@]}"; do
#             echo "Writing to file:" $f
            echo $f >> ../../out/"$TAG"_$i.txt
        done
    done    

    # Copy remaining vectors to txt file for this fold
    echo "Writing vectors for fold" $i
    for f in "${a[@]}"; do
#         echo "Writing to file:" $f
        echo $f >> ../../out/"$TAG"_$N.txt
    done

done


# Run MALLET classification experiments
# Do this for each of N folds
for((f=1;f<N+1;++f)); do
    echo "Running experiment; testing on fold" $f

    # Clear the appropriate train and test txt files
    >../../out/"$TAG"_"$f"_train.txt
    >../../out/"$TAG"_"$f"_test.txt
    
    # Write training data to train txt file
    for((i=1;i<f;++i)); do
        echo "Writing to file training fold:" $i
        cat ../../out/"$TAG"_$i.txt >> ../../out/"$TAG"_"$f"_train.txt
    done

    for((i=f+1;i<N+1;++i)); do
        echo "Writing to file training fold:" $i
        cat ../../out/"$TAG"_$i.txt >> ../../out/"$TAG"_"$f"_train.txt
    done

    # Write test data to test txt file
    echo "Writing to file testing fold:" $f
    cat ../../out/"$TAG"_$f.txt >> ../../out/"$TAG"_"$f"_test.txt

    # Convert the vector txt files to svm light files
    mallet import-svmlight --input ../../out/"$TAG"_"$f"_train.txt --output ../../out/"$TAG"_"$f"_train.vectors
    mallet import-svmlight --input ../../out/"$TAG"_"$f"_test.txt --output ../../out/"$TAG"_"$f"_test.vectors --use-pipe-from ../../out/"$TAG"_"$f"_train.vectors

    # Run MALLET and write classifier to txt file
    vectors2classify --training-file ../../out/"$TAG"_"$f"_train.vectors --testing-file ../../out/"$TAG"_"$f"_test.vectors --trainer "MaxEnt" --output-classifier ../../out/"$TAG"_"$f"_maxent
    classifier2info --classifier ../../out/"$TAG"_"$f"_maxent > ../../out/"$TAG"_"$f"_maxent.txt

done