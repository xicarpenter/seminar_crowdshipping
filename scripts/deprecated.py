from scripts.implementation import test_seed, test_seeds, CrowdshippingModel
import pickle


def compare_3_5(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int, 
               of: str = "MAX_PARCELS",
               seed: int = None,
               number_of_seeds: int = 1,
               print_level: int = 0):
    if number_of_seeds == 1:
        if seed is None:
            seeds, model_3_5 = test_seeds(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=True,
                                number_of_seeds=number_of_seeds)
            
            seed = seeds[0]
            model_3_5 = model_3_5[0]
            
            model_no_3_5 = test_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=False)
            
            print(f"--- Seed: {seed} ---")
            if of == "MAX_PROFIT":
                of_3_5 = model_3_5.calc_max_profit(print_level=0)
                of_no_3_5 = model_no_3_5.calc_max_profit(print_level=0)

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            elif of == "MAX_PARCELS":
                of_3_5 = model_3_5.calc_max_parcels()
                of_no_3_5 = model_no_3_5.calc_max_parcels()

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            else:
                raise ValueError("of must be MAX_PROFIT or MAX_PARCELS")

        else:
            model_3_5 = test_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=True)
            
            model_no_3_5 = test_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=False)
            
            with open(f"output/params.pkl", "wb") as f:
                pickle.dump(model_3_5._params, f)
            
            print(f"--- Seed: {seed} ---")
            if of == "MAX_PROFIT":
                of_3_5 = model_3_5.calc_max_profit(print_level=0)
                of_no_3_5 = model_no_3_5.calc_max_profit(print_level=0)

                print()
                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")
            
            elif of == "MAX_PARCELS":
                of_3_5 = model_3_5.calc_max_parcels()
                of_no_3_5 = model_no_3_5.calc_max_parcels()

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            else:
                raise ValueError("of must be MAX_PROFIT or MAX_PARCELS")    
    
    else:
        used_seeds, models_3_5 = test_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of=of,
                              number_of_seeds=number_of_seeds,
                              use_3_5=True)
        
        _, models_no_3_5 = test_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of=of,
                              number_of_seeds=number_of_seeds,
                              use_3_5=False,
                              used_seeds=used_seeds)
        
        for model_3_5, model_no_3_5 in zip(models_3_5, models_no_3_5):
            print(f"--- Seed: {used_seeds[models_3_5.index(model_3_5)]} ---")
            if of == "MAX_PROFIT":
                of_3_5 = model_3_5.calc_max_profit(print_level=0)
                of_no_3_5 = model_no_3_5.calc_max_profit(print_level=0)

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")
            
            elif of == "MAX_PARCELS":
                of_3_5 = model_3_5.calc_max_parcels()
                of_no_3_5 = model_no_3_5.calc_max_parcels()

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            else:
                raise ValueError("of must be MAX_PROFIT or MAX_PARCELS")
            