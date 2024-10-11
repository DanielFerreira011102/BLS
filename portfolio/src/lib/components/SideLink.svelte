<script lang="ts">
	export let href = '';
	export let title = '';
	export let subtitle = '';
	export let className = '';
</script>

<li
	class={`relative block border-t border-solid border-white border-opacity-5 bg-black bg-opacity-20 leading-7 ${className}`}
>
	<a {href} class="block py-8 pl-12 pr-32 uppercase tracking-widest text-white" on:click>
		{title}
		<span class="block lowercase leading-6 tracking-normal text-white text-opacity-50">
			{subtitle}
		</span>
	</a>
</li>
