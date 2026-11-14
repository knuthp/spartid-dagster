import dagster as dg


@dg.schedule(cron_schedule="@daily", job_name="all_assets_job")
def schedules(context: dg.ScheduleEvaluationContext) -> dg.RunRequest | dg.SkipReason:
    return dg.RunRequest(job_name="all_assets_job")