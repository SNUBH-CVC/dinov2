import submitit

# Replace with your actual log folder and job ID
job_folder = "./outputs"  # Folder where logs are stored
job_id = "1313"  # Ensure job_id is a string

# Retrieve the Job object using Job.fetch
job = submitit.Job(job_id=job_id, folder=job_folder)
job.cancel()