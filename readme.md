# Step 1
 ```console
     python ddm4ip/main.py exp=step1_argolight_cropped
 ```






# Step 2

   ```console
     python ddm4ip/main.py exp=step2_argolight_cropped exp_name="step2_lambda1_${l1}_lambda2_${l2}"   loss.lambda1=$l1 loss.lambda2=$l2
   ```

