import random
import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

def dp_common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    
    out = latent.copy()
    out["samples"] = samples
    return (out, )



class DP_Advanced_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "batch_mode": (["normal", "batch_all_random"], {"default": "normal"}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "enable_upscale": ("BOOLEAN", {"default": False}),
                    "two_step_upscale": ("BOOLEAN", {"default": False}),
                    "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),
                    "upscale_steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                    "upscale_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "upscale_sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "upscale_scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "upscale_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                }}

    RETURN_TYPES = ("LATENT", "IMAGE", "LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("sampled_latent", "sampled_image", "upscaled_latent", "upscaled_image", "sampler_info")
    FUNCTION = "sample"
    CATEGORY = "Desert Pixel/sampling"

    def sample(self, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, 
               latent_image, denoise, batch_mode, enable_upscale, two_step_upscale, scale_by, upscale_steps, upscale_cfg, 
               upscale_sampler_name, upscale_scheduler, upscale_denoise):
        
        # Handle batch mode
        latent_samples = latent_image["samples"]
        batch_size = latent_samples.shape[0]
        
        if batch_mode == "batch_all_random" and batch_size > 1:
            # Generate random seeds for each item in the batch
            random_seeds = [random.randint(0, 0xffffffffffffffff) for _ in range(batch_size)]
            
            # Pre-generate unique noise for each batch item with different seeds
            batch_noise_list = []
            for i, item_seed in enumerate(random_seeds):
                # Generate noise for each individual latent using its unique seed
                single_latent_sample = latent_samples[i:i+1]
                single_latent_sample = comfy.sample.fix_empty_latent_channels(model, single_latent_sample)
                item_noise = comfy.sample.prepare_noise(single_latent_sample, item_seed, None)
                batch_noise_list.append(item_noise)
            
            # Combine all unique noise into single batch tensor
            combined_noise = torch.cat(batch_noise_list, dim=0)
            
            # Prepare batch latent with proper structure
            batch_latent_for_sampling = latent_image.copy()
            
            # Handle noise mask if present
            noise_mask = None
            if "noise_mask" in latent_image:
                noise_mask = latent_image["noise_mask"]
            
            # Process entire batch at once with combined unique noise
            callback = latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            
            samples = comfy.sample.sample(model, combined_noise, steps, cfg, sampler_name, scheduler, 
                                        positive, negative, latent_samples,
                                        denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                        force_full_denoise=False, noise_mask=noise_mask, callback=callback, 
                                        disable_pbar=disable_pbar, seed=random_seeds[0])
            
            sampled_latent = latent_image.copy()
            sampled_latent["samples"] = samples
            seeds_info = f"Random seeds used for batch items: {random_seeds}"
        else:
            # Normal mode - use same seed for entire batch
            sampled_latent = dp_common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                              positive, negative, latent_image, denoise=denoise)[0]
            seeds_info = f"Seed used: {seed}"
        
        # Format sampling information with more detail
        info = (f"Initial Sampling:\n"
                f"  {seeds_info}\n"
                f"  Batch Size: {batch_size}\n"
                f"  Batch Mode: {batch_mode}\n"
                f"  Steps: {steps}\n"
                f"  CFG: {cfg:.1f}\n"
                f"  Sampler: {sampler_name}\n"
                f"  Scheduler: {scheduler}\n"
                f"  Denoise: {denoise:.2f}\n")
        
        # Decode the initial result
        sampled_image = vae.decode(sampled_latent["samples"])

        upscaled_latent = None
        upscaled_image = None

        # Handle upscaling if enabled
        if enable_upscale:
            info += (f"\nUpscaling:\n"
                    f"  Scale Factor: {scale_by:.2f}x\n"
                    f"  Method: bicubic\n"
                    f"  Two-step: {'Yes' if two_step_upscale else 'No'}\n")

            to_upscale = sampled_latent.copy()

            if two_step_upscale:
                # Calculate intermediate scale factor (square root for geometric progression)
                mid_scale = scale_by ** 0.5
                
                info += (f"  Step 1:\n"
                        f"    Scale: {mid_scale:.2f}x\n"
                        f"    Seed: {seed + 1}\n"
                        f"    Steps: {upscale_steps}\n"
                        f"    CFG: {upscale_cfg:.1f}\n"
                        f"    Sampler: {upscale_sampler_name}\n"
                        f"    Scheduler: {upscale_scheduler}\n"
                        f"    Denoise: {upscale_denoise:.2f}\n")

                # First upscale
                width = round(to_upscale["samples"].shape[-1] * mid_scale)
                height = round(to_upscale["samples"].shape[-2] * mid_scale)
                s = to_upscale.copy()
                s["samples"] = comfy.utils.common_upscale(to_upscale["samples"], width, height, "bicubic", "disabled")
                
                # First refinement
                mid_latent = dp_common_ksampler(model, seed + 1, upscale_steps, upscale_cfg, 
                                              upscale_sampler_name, upscale_scheduler, 
                                              positive, negative, s, denoise=upscale_denoise)[0]

                # Increase denoise by 0.05 for second pass
                second_pass_denoise = min(1.0, upscale_denoise + 0.05)

                info += (f"  Step 2:\n"
                        f"    Scale: {mid_scale:.2f}x\n"
                        f"    Seed: {seed + 2}\n"
                        f"    Steps: {upscale_steps}\n"
                        f"    CFG: {upscale_cfg:.1f}\n"
                        f"    Sampler: {upscale_sampler_name}\n"
                        f"    Scheduler: {upscale_scheduler}\n"
                        f"    Denoise: {second_pass_denoise:.2f}\n")

                # Second upscale
                width = round(mid_latent["samples"].shape[-1] * mid_scale)
                height = round(mid_latent["samples"].shape[-2] * mid_scale)
                s = mid_latent.copy()
                s["samples"] = comfy.utils.common_upscale(mid_latent["samples"], width, height, "bicubic", "disabled")
                
                # Final refinement with increased denoise
                upscaled_latent = dp_common_ksampler(model, seed + 2, upscale_steps, upscale_cfg, 
                                                   upscale_sampler_name, upscale_scheduler, 
                                                   positive, negative, s, denoise=second_pass_denoise)[0]
            else:
                # Original single-step upscale
                info += (f"  Single Step:\n"
                        f"    Seed: {seed + 1}\n"
                        f"    Steps: {upscale_steps}\n"
                        f"    CFG: {upscale_cfg:.1f}\n"
                        f"    Sampler: {upscale_sampler_name}\n"
                        f"    Scheduler: {upscale_scheduler}\n"
                        f"    Denoise: {upscale_denoise:.2f}\n")

                width = round(to_upscale["samples"].shape[-1] * scale_by)
                height = round(to_upscale["samples"].shape[-2] * scale_by)
                s = to_upscale.copy()
                s["samples"] = comfy.utils.common_upscale(to_upscale["samples"], width, height, "bicubic", "disabled")
                
                upscaled_latent = dp_common_ksampler(model, seed + 1, upscale_steps, upscale_cfg, 
                                                   upscale_sampler_name, upscale_scheduler, 
                                                   positive, negative, s, denoise=upscale_denoise)[0]
            
            upscaled_image = vae.decode(upscaled_latent["samples"])

        return (sampled_latent, sampled_image, upscaled_latent, upscaled_image, info)