<script lang="ts">
    import { Icon } from 'svelte-icons-pack';
    import { RiSystemDownloadLine } from 'svelte-icons-pack/ri';

    export let document: {
        title: string;
        src: string;
        image: string;
        download: boolean;
        type: string;
        description: string;
    };
</script>

<a
    target="_blank"
    rel="noopener noreferrer"
    href={document.download ? document.src : undefined}
    download={document.title}
    class="group relative flex h-full flex-col overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm transition-all hover:shadow-md"
>
    <!-- Icon Container -->
    <div class="relative flex h-48 items-center justify-center bg-gray-50">
        <img 
            src={document.image}
            alt="presentation" 
            class="h-28 w-28 transition-transform duration-300 group-hover:scale-105" 
        />
    </div>

    <!-- Content -->
    <div class="flex items-center justify-between p-4">
        <h3 class="flex-1 text-base font-semibold leading-tight text-gray-900 line-clamp-2">
            {document.title}
        </h3>
        {#if document.download}
            <div class="ml-3 flex-shrink-0">
                <Icon 
                    src={RiSystemDownloadLine} 
                    size="24" 
                    className="text-gray-400 group-hover:text-gray-600 transition-colors duration-200" 
                />
            </div>
        {/if}
    </div>
</a>