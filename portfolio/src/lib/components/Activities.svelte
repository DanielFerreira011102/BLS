<script lang="ts">
    import { dataStore } from '$lib/stores/dataStore';
    import ActivityCard from '$lib/components/ActivityCard.svelte';

    let activities = $dataStore.ACTIVITIES;
</script>

<h2 class="m-0 mb-12 font-mono text-2xl font-semibold uppercase tracking-widest text-black">
    /Activities
</h2>
<div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 2xl:grid-cols-4 gap-y-8 gap-x-8">
    {#each activities as activity}
        <ActivityCard {activity} />
    {/each}
</div>