#!/bin/sh

# First argument to script is run tag
TAG=$1

# Second argument to script is number of folds for n-fold cross-validation
N=$2


# Build the feature vectors from the input sentences
./build_vectors.py ../../data/sentences.txt


# Split vectors randomly into N folds, balanced by class
# The size of each fold for the 1st class is the length of the class1 array / N
K1=$((${#class1[*]} / N))
echo "Each fold for the first class will have this many vectors:" $K1

# The size of each fold for the 2nd class is the length of the class2 array / N
K2=$((${#class2[*]} / N))
echo "Each fold for the second class will have this many vectors:" $K2

# Get 1/Nth of the vectors from each class at random
# Do this N-1 times
for((i=1;i<N && ${#class1[@]};++i)); do
    echo "Writing vectors for fold" $i

    # Create an array to hold N randomly selected files
    randf=()
    
    # Do this K1 times
    for((k=0;k<K1 && ${#class1[@]};++k)); do
        # Pick random file
        ((j=RANDOM%${#class1[@]}))
        # Add to the array of random files
        randf+=( "${class1[j]}" )
        # Remove from the array of all files
        class1=( "${class1[@]:0:j}" "${class1[@]:j+1}" )
    done

    # Do this K2 times
    for((k=0;k<K2 && ${#class2[@]};++k)); do
        # Pick random file
        ((j=RANDOM%${#class2[@]}))
        # Add to the array of random files
        randf+=( "${class2[j]}" )
        # Remove from the array of all files
        class2=( "${class2[@]:0:j}" "${class2[@]:j+1}" )
    done

    # Append random vector files to blank txt file
    > ../out/"$TAG"_$i.txt
    for f in "${randf[@]}"; do
        echo "Writing to file:" $f
        cat $f >> ../out/"$TAG"_$i.txt
    done
done

# Copy remaining vectors to blank txt file
> ../out/"$TAG"_$N.txt
echo "Writing vectors for fold" $i
for f in "${class1[@]}"; do
    echo "Writing to file:" $f
    cat $f >> ../out/"$TAG"_$N.txt
done

for f in "${class2[@]}"; do
    echo "Writing to file:" $f
    cat $f >> ../out/"$TAG"_$N.txt
done


# Run MALLET classification experiments
# Do this for each of N folds
for((f=1;f<N+1;++f)); do
    echo "Running experiment; testing on fold" $f

    # Clear the appropriate train and test txt files
    >../out/"$TAG"_"$f"_train.txt
    >../out/"$TAG"_"$f"_test.txt
    
    # Write training data to train txt file
    for((i=1;i<f;++i)); do
        echo "Writing to file training fold:" $i
        cat ../out/"$TAG"_$i.txt >> ../out/"$TAG"_"$f"_train.txt
    done

    for((i=f+1;i<N+1;++i)); do
        echo "Writing to file training fold:" $i
        cat ../out/"$TAG"_$i.txt >> ../out/"$TAG"_"$f"_train.txt
    done

    # Write test data to test txt file
    echo "Writing to file testing fold:" $f
    cat ../out/"$TAG"_$f.txt >> ../out/"$TAG"_"$f"_test.txt

    # Convert the vector txt files to svm light files
    mallet import-svmlight --input ../out/"$TAG"_"$f"_train.txt --output ../out/"$TAG"_"$f"_train.vectors
    mallet import-svmlight --input ../out/"$TAG"_"$f"_test.txt --output ../out/"$TAG"_"$f"_test.vectors --use-pipe-from ../out/"$TAG"_"$f"_train.vectors

    # Run MALLET and write classifier to txt file
    vectors2classify --training-file ../out/"$TAG"_"$f"_train.vectors --testing-file ../out/"$TAG"_"$f"_test.vectors --trainer "MaxEnt" --output-classifier ../out/"$TAG"_"$f"_maxent
    classifier2info --classifier ../out/"$TAG"_"$f"_maxent > ../out/"$TAG"_"$f"_maxent.txt

done