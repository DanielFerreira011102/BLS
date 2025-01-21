<script lang="ts">
    export let activity: {
        title: string;
        image: string;
        src?: string;
        description?: string;
        download?: boolean;
        tags: string[];
        date: string;
    };

    // Format date from YYYY-MM-DD to Month DD, YYYY
    function formatDate(dateStr: string): string {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }

    function handleClick() {
        if (activity.src && activity.download) {
            // Create a temporary link element
            const link = document.createElement('a');
            link.href = activity.src;
            link.download = ''; // This will use the original filename
            link.target = '_blank';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
</script>

<div 
    class="group relative flex h-full flex-col overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm transition-all hover:shadow-md cursor-pointer"
    on:click={handleClick}
    on:keydown={(e) => e.key === 'Enter' && handleClick()}
    role="button"
    tabindex="0"
>
    <!-- Image -->
    <div class="relative h-48 overflow-hidden">
        <img
            src={activity.image}
            alt={activity.title}
            class="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
        />
    </div>

    <!-- Content -->
    <div class="flex flex-1 flex-col p-4">
        <!-- Tags -->
        <div class="mb-2 flex flex-wrap gap-2">
            {#each activity.tags as tag}
                <span class="inline-flex items-center rounded-full bg-blue-100 px-3 py-0.5 text-sm font-medium text-blue-800">
                    {tag}
                </span>
            {/each}
        </div>

        <!-- Title -->
        <h3 class="mb-2 flex-1 text-lg font-semibold leading-tight text-gray-900">
            {activity.title}
        </h3>

        <!-- Date -->
        <div class="mt-2 text-sm text-gray-600">
            {formatDate(activity.date)}
        </div>
    </div>
</div>