<script>
    export let meta;
    export let personal;
    export let contact;
</script>

<footer id="footer">
    <div class="container mx-auto">
        <div class="border-t border-gray-300 my-9"></div>
        <div class="flex justify-between items-center">
            <div class="text-left">
                <p>© {meta.creation_year} /
                    <a href={contact.github} class="text-gray-600 no-underline">
                        {personal.name}
                    </a>
                </p>
            </div>
            <div class="text-right">
                <a class="text-gray-600 no-underline" href={meta.src}>View Source</a>
            </div>
        </div>
    </div>
</footer>