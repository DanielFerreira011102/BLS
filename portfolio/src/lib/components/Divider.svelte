<script>
	export let color = 'gray-300';
	export let margin = '0';
	export let height = 'px';
	export let className = '';
</script>

<div class={`my-${margin} bg-${color} h-${height} ${className}`}></div>
